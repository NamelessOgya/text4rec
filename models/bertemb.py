from .base import BaseModel
from .bert_modules.bertemb import BERTEmbeddingTransformer
import torch
import torch.nn as nn
import numpy as np


class BERTEmbeddingModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERTEmbeddingTransformer(args)
        
        # Load pre-trained item embeddings
        item_embeddings = torch.from_numpy(np.load(args.item_embedding_path)).float()
        
        # Add a zero vector for padding (item_id 0)
        padding_embedding = torch.zeros(1, item_embeddings.size(1))
        self.item_embeddings = nn.Parameter(torch.cat([padding_embedding, item_embeddings], dim=0))
        # To make embeddings trainable or not
        # self.item_embeddings.requires_grad = False # if you want to keep it fixed

        # Add a prediction layer to project transformer output to the same space as item embeddings
        self.prediction_layer = nn.Linear(args.bert_hidden_units, args.bert_hidden_units)


    @classmethod
    def code(cls):
        return 'bert_embedding'

    def forward(self, x):
        # x is expected to be a sequence of embeddings (batch_size, seq_len, embedding_dim)
        x = self.bert(x)
        
        # Project the output of the transformer
        x_projected = self.prediction_layer(x)

        # Return the sequence of vectors, not the final logits
        return x_projected
