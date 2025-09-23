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
            # Now InfoNCE also uses explicit negatives, so batch should have 4 elements
            input_sequences, positive_embeddings, mask_indices, negative_embeddings = batch

            sequence_output = self.model(input_sequences)

            mask_indices = mask_indices.squeeze()
            batch_indices = torch.arange(sequence_output.size(0), device=sequence_output.device)
            anchor_embeddings = sequence_output[batch_indices, mask_indices]
            
            projected_positive_embeddings = model.projection_layer(positive_embeddings)

            anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
            projected_positive_embeddings = F.normalize(projected_positive_embeddings, p=2, dim=1)

            if self.args.loss_type == 'infonce':
                # 1. In-batch similarity matrix
                in_batch_sim_matrix = torch.matmul(anchor_embeddings, projected_positive_embeddings.t()) / self.temperature

                # 2. Explicit negative scores
                explicit_negative_scores = torch.tensor([], device=anchor_embeddings.device)
                if negative_embeddings is not None and negative_embeddings.numel() > 0:
                    negative_embeddings = negative_embeddings.to(anchor_embeddings.device)
                    projected_negative_embeddings = model.projection_layer(negative_embeddings)
                    projected_negative_embeddings = F.normalize(projected_negative_embeddings, p=2, dim=1)
                    # Corrected matmul for explicit negatives
                    explicit_negative_scores = torch.matmul(anchor_embeddings.unsqueeze(1), projected_negative_embeddings.transpose(1, 2)).squeeze(1) / self.temperature

                # 3. Combine scores
                positive_scores = in_batch_sim_matrix.diag().unsqueeze(1)
                mask = torch.eye(in_batch_sim_matrix.size(0), device=in_batch_sim_matrix.device).bool()
                in_batch_negative_scores = in_batch_sim_matrix.masked_fill(mask, -float('inf'))

                if self.use_hard_negative_mining:
                    full_negative_pool = torch.cat([in_batch_negative_scores, explicit_negative_scores], dim=1)
                    batch_size = full_negative_pool.size(0)
                    if batch_size <= 1:
                        return torch.tensor(0.0, device=in_batch_sim_matrix.device, requires_grad=True)

                    if self.args.use_curriculum_learning:
                        # Refined Curriculum: Keep total negatives constant, vary the mix of hard/easy.
                        total_negatives_in_loss = batch_size - 1
                        
                        k_initial = self.args.hard_negative_curriculum_k_initial
                        k_final = self.args.hard_negative_curriculum_k_final
                        total_epochs = self.args.hard_negative_curriculum_total_epochs
                        
                        if self.epoch < total_epochs:
                            num_hard = k_initial + int((k_final - k_initial) * (self.epoch / total_epochs))
                        else:
                            num_hard = k_final
                        
                        num_hard = min(num_hard, total_negatives_in_loss)
                        num_easy = total_negatives_in_loss - num_hard

                        hard_negative_scores, _ = full_negative_pool.topk(num_hard, dim=1)
                        
                        easy_negative_scores = torch.empty(batch_size, 0, device=in_batch_sim_matrix.device)
                        if num_easy > 0:
                            # To get easy negatives, we sort the pool and take the easiest ones that are not the hard ones
                            # A simpler proxy is to take the easiest overall from the combined pool
                            easy_negative_scores, _ = full_negative_pool.topk(num_easy, dim=1, largest=False)

                        logits = torch.cat([positive_scores, hard_negative_scores, easy_negative_scores], dim=1)

                    else:
                        # Original hard negative mining (single hardest from the combined pool)
                        hardest_negative_scores, _ = full_negative_pool.max(dim=1, keepdim=True)
                        logits = torch.cat([positive_scores, hardest_negative_scores], dim=1)

                else: # Standard InfoNCE with combined negatives
                    all_scores = torch.cat([in_batch_sim_matrix, explicit_negative_scores], dim=1)
                    # The diagonal of in_batch_sim_matrix corresponds to the positive scores
                    # So we can use the full matrix and set the labels to 0
                    logits = all_scores

                labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
                loss = F.cross_entropy(logits, labels)
            
            else: # bce or gbce
                pos_logits = (anchor_embeddings * projected_positive_embeddings).sum(dim=1)

                # Explicit negatives ONLY
                if negative_embeddings is not None and negative_embeddings.numel() > 0:
                    negative_embeddings = negative_embeddings.to(anchor_embeddings.device)
                    projected_negative_embeddings = model.projection_layer(negative_embeddings)
                    projected_negative_embeddings = F.normalize(projected_negative_embeddings, p=2, dim=1)
                    neg_logits = (anchor_embeddings.unsqueeze(1) * projected_negative_embeddings).sum(dim=-1)
                else:
                    # This path should not be taken if using bce/gbce with negative sampling, but as a safeguard:
                    neg_logits = torch.empty(anchor_embeddings.size(0), 0, device=anchor_embeddings.device)

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

                    # More stable gBCE calculation
                    pos_labels = torch.ones_like(pos_logits)
                    neg_labels = torch.zeros_like(neg_logits)
                    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)
                    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
                    loss = (pos_loss * beta) + neg_loss
            
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
            
            # Get the hidden state from the BERT model, before the large output layer.
            bert_output = model.bert(seqs) # (B, L, H)
            last_hidden_state = bert_output[:, -1, :] # (B, H)

            # Get the weights and biases from the final output layer.
            output_layer = model.out # nn.Linear(H, V)
            candidate_weights = output_layer.weight[candidates] # (B, C, H)
            candidate_biases = output_layer.bias[candidates] # (B, C)

            # Calculate scores only for the candidate items.
            # Reshape last_hidden_state for broadcasting: (B, 1, H)
            last_hidden_state = last_hidden_state.unsqueeze(1)
            
            # Perform batch matrix multiplication: (B, 1, H) @ (B, H, C) -> (B, 1, C)
            scores = torch.bmm(last_hidden_state, candidate_weights.transpose(1, 2)).squeeze(1)
            scores += candidate_biases # Add biases
            
            metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
            return metrics
