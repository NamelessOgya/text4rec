from .base import BaseModel
from .bert_modules.bert import BERT, BERTEmbeddingTransformer
import torch
import torch.nn as nn
import numpy as np


class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)


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

    @classmethod
    def code(cls):
        return 'bert_embedding'

    def forward(self, x):
        # x is expected to be a sequence of embeddings (batch_size, seq_len, embedding_dim)
        x = self.bert(x)
        
        # Calculate logits by taking the dot product with all item embeddings
        # x shape: (batch_size, seq_len, hidden_dim)
        # self.item_embeddings shape: (num_items + 1, hidden_dim)
        # logits shape: (batch_size, seq_len, num_items + 1)
        logits = torch.matmul(x, self.item_embeddings.transpose(0, 1))
        return logits
