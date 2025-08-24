from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
import torch
import torch.nn as nn


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

        # Calculate the triplet loss
        loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        return loss

    def calculate_metrics(self, batch):
        # The evaluation logic remains the same as it uses the evaluation dataloader
        # which provides pre-defined candidates and labels for ranking metrics.
        seqs, candidates, labels = batch
        
        # The model's forward pass returns a sequence of vectors
        vectors = self.model(seqs)  # B x T x D
        
        # We are interested in the vector for the last item (the one to be predicted)
        last_vector = vectors[:, -1, :]  # B x D
        
        # Manually compute scores (logits) by taking the dot product with all item embeddings
        # Note: self.model.item_embeddings is available because the model is BERTEmbeddingModel
        all_item_embeddings = self.model.item_embeddings
        if self.is_parallel: # if model is DataParallel
            all_item_embeddings = self.model.module.item_embeddings

        logits = torch.matmul(last_vector, all_item_embeddings.transpose(0, 1)) # B x V

        # Gather the scores for the specific candidates provided for evaluation
        scores = logits.gather(1, candidates)  # B x C

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics
