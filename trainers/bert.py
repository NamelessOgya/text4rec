from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        # Use TripletMarginLoss for metric learning, with a margin of 1.0
        self.loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

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
        # Unpack the batch from the new dataloader for triplet loss
        input_sequences, positive_embeddings, negative_embeddings, mask_indices = batch
        
        # The model's forward pass returns the transformer's output for the whole sequence
        sequence_output = self.model(input_sequences)  # B x T x D

        # Extract the anchor embedding at the masked position for each sequence in the batch
        # mask_indices is B x 1, needs to be squeezed to use with torch.arange
        mask_indices = mask_indices.squeeze()
        # Create an index for the batch dimension
        batch_indices = torch.arange(sequence_output.size(0), device=sequence_output.device)
        # Gather the anchor embeddings using the batch and mask indices
        anchor_embeddings = sequence_output[batch_indices, mask_indices]

        # Get the underlying model if using DataParallel
        model = self.model.module if self.is_parallel else self.model
        
        # Project the positive and negative embeddings
        projected_positive_embeddings = model.projection_layer(positive_embeddings)
        projected_negative_embeddings = model.projection_layer(negative_embeddings)

        # L2-normalize all embeddings before feeding them to the loss function
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        projected_positive_embeddings = F.normalize(projected_positive_embeddings, p=2, dim=1)
        projected_negative_embeddings = F.normalize(projected_negative_embeddings, p=2, dim=1)

        # Calculate the triplet loss
        loss = self.loss_fn(anchor_embeddings, projected_positive_embeddings, projected_negative_embeddings)
        return loss

    def calculate_metrics(self, batch):
        # The evaluation logic remains the same as it uses the evaluation dataloader
        # which provides pre-defined candidates and labels for ranking metrics.
        seqs, candidates, labels = batch
        
        # The model's forward pass returns a sequence of vectors
        vectors = self.model(seqs)  # B x T x D
        
        # We are interested in the vector for the last item (the one to be predicted)
        last_vector = vectors[:, -1, :]  # B x D
        
        # Get the underlying model if using DataParallel
        model = self.model.module if self.is_parallel else self.model

        # Project all item embeddings before calculating scores
        projected_all_item_embeddings = model.projection_layer(model.item_embeddings)

        # L2-normalize vectors for consistent scoring
        last_vector = F.normalize(last_vector, p=2, dim=1)
        projected_all_item_embeddings = F.normalize(projected_all_item_embeddings, p=2, dim=1)

        logits = torch.matmul(last_vector, projected_all_item_embeddings.transpose(0, 1)) # B x V

        # Gather the scores for the specific candidates provided for evaluation
        scores = logits.gather(1, candidates)  # B x C

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics