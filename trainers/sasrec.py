from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
import torch
import torch.nn as nn
import torch.nn.functional as F

class SASTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.temperature = args.infonce_temperature

    @classmethod
    def code(cls):
        return 'sasrec'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        model = self.model.module if self.is_parallel else self.model
        seqs, labels = batch

        # (B, S, H)
        output_vectors = self.model(seqs)

        # Reshape for vectorized processing
        # (B*S, H)
        anchor_embeddings = output_vectors.view(-1, output_vectors.size(-1))
        # (B*S)
        positive_ids = labels.view(-1)

        # Filter out padding
        padding_mask = positive_ids != 0
        anchor_embeddings = anchor_embeddings[padding_mask]
        positive_ids = positive_ids[padding_mask]

        if len(positive_ids) == 0:
            return torch.tensor(0.0, device=seqs.device, requires_grad=True)

        # Get positive embeddings and project them
        positive_embeddings_orig = model.item_embeddings[positive_ids]
        positive_embeddings = model.projection_layer(positive_embeddings_orig)

        # Vectorized negative sampling
        num_neg_samples = self.args.train_negative_sample_size
        num_items = self.train_loader.dataset.item_count
        
        neg_indices = torch.randint(1, num_items + 1, 
                                    (positive_ids.size(0), num_neg_samples), 
                                    device=seqs.device)

        # Get negative embeddings and project them
        negative_embeddings_orig = model.item_embeddings[neg_indices]
        orig_dim = negative_embeddings_orig.size(-1)
        negative_embeddings = model.projection_layer(negative_embeddings_orig.view(-1, orig_dim))
        negative_embeddings = negative_embeddings.view(positive_ids.size(0), num_neg_samples, -1)

        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=2)

        # Calculate logits
        pos_logits = (anchor_embeddings * positive_embeddings).sum(dim=-1)
        neg_logits = (anchor_embeddings.unsqueeze(1) * negative_embeddings).sum(dim=-1)

        if self.args.loss_type == 'bce':
            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
            loss = (pos_loss + neg_loss) / 2
        elif self.args.loss_type == 'gbce':
            negs_per_pos = self.args.train_negative_sample_size
            alpha = negs_per_pos / (num_items - 1)
            t = self.args.gbce_q
            beta = alpha * ((1 - 1/alpha)*t + 1/alpha)

            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
            loss = ((pos_loss * beta) + neg_loss) / (beta + 1)
        else: # Default to InfoNCE
            all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
            all_logits /= self.temperature
            labels_infonce = torch.zeros(all_logits.size(0), dtype=torch.long, device=all_logits.device)
            loss = F.cross_entropy(all_logits, labels_infonce)
        
        return loss

    def calculate_metrics(self, batch):
        model = self.model.module if self.is_parallel else self.model
        seqs, candidates, labels = batch
        
        vectors = self.model(seqs)
        last_vector = vectors[:, -1, :] # Get the last hidden state
        
        # Project all item embeddings to the same space as the model output
        projected_all_item_embeddings = model.projection_layer(model.item_embeddings)

        # Normalize for cosine similarity
        last_vector = F.normalize(last_vector, p=2, dim=1)
        projected_all_item_embeddings = F.normalize(projected_all_item_embeddings, p=2, dim=1)

        # Calculate scores against all items
        logits = torch.matmul(last_vector, projected_all_item_embeddings.transpose(0, 1))
        
        # Gather scores for candidate items
        scores = logits.gather(1, candidates)

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics
