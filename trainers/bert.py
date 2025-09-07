from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        # Define loss function for InfoNCE
        self.temperature = 0.07  # Temperature for scaling logits
        self.loss_fn = nn.CrossEntropyLoss()

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
            # Unpack the batch from the metric learning dataloader
            input_sequences, positive_embeddings, mask_indices = batch
            
            # The model's forward pass returns the transformer's output for the whole sequence
            sequence_output = self.model(input_sequences)  # B x T x D

            # Extract the anchor embedding at the masked position
            mask_indices = mask_indices.squeeze()
            batch_indices = torch.arange(sequence_output.size(0), device=sequence_output.device)
            anchor_embeddings = sequence_output[batch_indices, mask_indices]
            
            # Project the positive embeddings
            projected_positive_embeddings = model.projection_layer(positive_embeddings)

            # L2-normalize all embeddings
            anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
            projected_positive_embeddings = F.normalize(projected_positive_embeddings, p=2, dim=1)

            # InfoNCE loss using in-batch negatives
            sim_matrix = torch.matmul(anchor_embeddings, projected_positive_embeddings.t()) / self.temperature
            
            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

            loss = self.loss_fn(sim_matrix, labels)
            return loss
        else:
            seqs, labels = batch
            logits = self.model(seqs) # B x T x V
            
            logits = logits.view(-1, logits.size(-1)) # (B*T) x V
            labels = labels.view(-1) # B*T
            
            loss = self.loss_fn(logits, labels)
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