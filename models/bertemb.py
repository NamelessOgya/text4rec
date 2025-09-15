from .base import BaseModel
from .bert_modules.bertemb import BERTEmbeddingTransformer
import torch
import torch.nn as nn
import numpy as np


class BERTEmbeddingModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        # Load pre-trained item embeddings to determine their dimension
        item_embeddings = torch.from_numpy(np.load(args.item_embedding_path)).float()
        embedding_dim = item_embeddings.size(1)

        self.pre_bert_mlp = nn.Sequential(
            nn.Linear(embedding_dim, args.bert_hidden_units),
            nn.ReLU(),
            nn.Linear(args.bert_hidden_units, args.bert_hidden_units)
        )
        self.bert = BERTEmbeddingTransformer(args)

        # --- Build the retrieval projection MLP dynamically ---
        if hasattr(args, 'projection_mlp_dims') and args.projection_mlp_dims:
            mlp_dims = args.projection_mlp_dims
            input_dim = args.bert_hidden_units
            all_dims = [input_dim] + mlp_dims

            layers = []
            for i in range(len(all_dims) - 2):
                layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
                layers.append(nn.ReLU())
                if hasattr(args, 'projection_dropout') and args.projection_dropout > 0:
                    layers.append(nn.Dropout(args.projection_dropout))
            
            layers.append(nn.Linear(all_dims[-2], all_dims[-1]))
            self.retrieval_projection = nn.Sequential(*layers)
        else:
            # Fallback to identity if no MLP is defined
            self.retrieval_projection = nn.Identity()

        padding_embedding = torch.zeros(1, embedding_dim)
        self.item_embeddings = nn.Parameter(torch.cat([padding_embedding, item_embeddings], dim=0))

        # Projection for item ID embeddings: mlp -> retrieval_projection
        self.projection_layer = nn.Sequential(self.pre_bert_mlp, self.retrieval_projection)


    @classmethod
    def code(cls):
        return 'bert_embedding'

    def forward(self, x):
        # x is expected to be a sequence of embeddings (batch_size, seq_len, embedding_dim)
        x = self.pre_bert_mlp(x)
        x = self.bert(x)
        x = self.retrieval_projection(x)

        # Return the sequence of vectors projected to the retrieval dimension
        return x