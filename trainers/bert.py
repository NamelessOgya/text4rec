from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.temperature = args.infonce_temperature
        self.use_hard_negative_mining = args.use_hard_negative_mining

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        model = self.model.module if self.is_parallel else self.model
        if hasattr(model, 'projection_layer'):
            if len(batch) == 4:
                input_sequences, positive_embeddings, mask_indices, negative_embeddings = batch
            else:
                input_sequences, positive_embeddings, mask_indices = batch
                negative_embeddings = None

            sequence_output = self.model(input_sequences)

            mask_indices = mask_indices.squeeze()
            batch_indices = torch.arange(sequence_output.size(0), device=sequence_output.device)
            anchor_embeddings = sequence_output[batch_indices, mask_indices]
            
            projected_positive_embeddings = model.projection_layer(positive_embeddings)

            anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
            projected_positive_embeddings = F.normalize(projected_positive_embeddings, p=2, dim=1)

            if self.args.loss_type == 'infonce':
                sim_matrix = torch.matmul(anchor_embeddings, projected_positive_embeddings.t()) / self.temperature
                
                if self.use_hard_negative_mining:
                    batch_size = sim_matrix.size(0)
                    mask = torch.eye(batch_size, device=sim_matrix.device).bool()
                    negative_scores = sim_matrix.masked_fill(mask, -float('inf'))
                    hardest_negative_scores, _ = negative_scores.max(dim=1, keepdim=True)
                    positive_scores = sim_matrix.diag().unsqueeze(1)
                    logits = torch.cat([positive_scores, hardest_negative_scores], dim=1)
                    labels = torch.zeros(batch_size, dtype=torch.long, device=sim_matrix.device)
                    loss = F.cross_entropy(logits, labels)
                else:
                    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
                    loss = F.cross_entropy(sim_matrix, labels)
            
            else: # bce or gbce
                pos_logits = (anchor_embeddings * projected_positive_embeddings).sum(dim=1)

                # In-batch negatives
                in_batch_scores = torch.matmul(anchor_embeddings, projected_positive_embeddings.t())
                mask = torch.eye(in_batch_scores.size(0), device=in_batch_scores.device).bool()
                neg_logits = in_batch_scores[~mask].view(in_batch_scores.size(0), -1)

                # Explicit negatives
                if negative_embeddings is not None and negative_embeddings.numel() > 0:
                    negative_embeddings = negative_embeddings.to(anchor_embeddings.device)
                    projected_negative_embeddings = model.projection_layer(negative_embeddings)
                    projected_negative_embeddings = F.normalize(projected_negative_embeddings, p=2, dim=1)
                    explicit_negative_scores = (anchor_embeddings.unsqueeze(1) * projected_negative_embeddings).sum(dim=-1)
                    neg_logits = torch.cat([neg_logits, explicit_negative_scores], dim=1)

                if self.args.loss_type == 'bce':
                    pos_labels = torch.ones_like(pos_logits)
                    neg_labels = torch.zeros_like(neg_logits)
                    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)
                    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
                    loss = pos_loss + neg_loss
                elif self.args.loss_type == 'gbce':
                    num_items = self.train_loader.dataset.num_items
                    negs_per_pos = self.args.train_negative_sample_size
                    alpha = negs_per_pos / (num_items - 1)
                    t = self.args.gbce_q
                    beta = alpha * ((1 - 1/alpha)*t + 1/alpha)

                    # transform positive logits
                    eps = 1e-10
                    positive_probs = torch.sigmoid(pos_logits)
                    positive_probs_g = torch.pow(positive_probs, beta)
                    positive_probs_g = torch.clamp(positive_probs_g, eps, 1 - eps)
                    pos_logits_transformed = torch.log(positive_probs_g / (1 - positive_probs_g))

                    # calculate loss
                    pos_labels = torch.ones_like(pos_logits_transformed)
                    neg_labels = torch.zeros_like(neg_logits)
                    pos_loss = F.binary_cross_entropy_with_logits(pos_logits_transformed, pos_labels)
                    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
                    loss = pos_loss + neg_loss
            
            return loss
        else:
            seqs, labels = batch
            logits = self.model(seqs) # B x T x V

            if self.args.dataloader_code == 'bert_prefix_augmentation':
                last_item_indices = (seqs != 0).sum(dim=1) - 1
                last_logits = logits[torch.arange(logits.size(0)), last_item_indices]
                last_labels = labels[torch.arange(labels.size(0)), last_item_indices]
                loss = F.cross_entropy(last_logits, last_labels, ignore_index=0)
            else: # Original BERT (Cloze task)
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                loss = F.cross_entropy(logits, labels, ignore_index=0)
            
            return loss

    def calculate_metrics(self, batch):
        model = self.model.module if self.is_parallel else self.model
        if hasattr(model, 'projection_layer'):
            seqs, candidates, labels = batch
            vectors = self.model(seqs)
            last_vector = vectors[:, -1, :]
            
            projected_all_item_embeddings = model.projection_layer(model.item_embeddings)

            last_vector = F.normalize(last_vector, p=2, dim=1)
            projected_all_item_embeddings = F.normalize(projected_all_item_embeddings, p=2, dim=1)

            logits = torch.matmul(last_vector, projected_all_item_embeddings.transpose(0, 1))
            scores = logits.gather(1, candidates)

            metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
            return metrics
        else:
            seqs, candidates, labels = batch
            logits = self.model(seqs)
            
            last_logits = logits[:, -1, :] # B x V
            scores = last_logits.gather(1, candidates) # B x C
            
            metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
            return metrics
