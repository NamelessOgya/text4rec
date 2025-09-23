from .base import BaseModel
from .bert_modules.transformer import TransformerBlock
from .bert_modules.embedding.position import PositionalEmbedding
import torch
import torch.nn as nn
import numpy as np


class SASRecModel(BaseModel):
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
        self.sasrec = SASRec(args)

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
        return 'sasrec'

    def forward(self, x):
        # If the input is a LongTensor of item IDs, look up the embeddings
        if x.dtype == torch.long:
            # Create a mask for padding tokens (ID 0)
            padding_mask = (x > 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
            x = self.item_embeddings[x] # Convert IDs to embeddings
        else:
            # If input is already embeddings, we can't know padding. Assume no padding.
            padding_mask = torch.ones(x.size(0), 1, 1, x.size(1), device=x.device, dtype=torch.bool)


        # x is now a sequence of embeddings (batch_size, seq_len, embedding_dim)
        x = self.pre_bert_mlp(x)
        x = self.sasrec(x, padding_mask)
        x = self.retrieval_projection(x)

        # Return the sequence of vectors projected to the retrieval dimension
        return x


class SASRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden = args.bert_hidden_units
        n_layers = args.bert_num_blocks
        attn_heads = args.bert_num_heads
        dropout = args.bert_dropout
        max_len = args.bert_max_len

        self.pos_emb = PositionalEmbedding(max_len=max_len, d_model=self.hidden)
        self.emb_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, attn_heads, self.hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, padding_mask):
        # x is (B, S, H)
        # Positional embedding
        x += self.pos_emb(x)[:, :x.size(1), :] # Slice to sequence length
        x = self.emb_dropout(x)

        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0) # (1, 1, S, S)
        
        # Combine with padding mask
        mask = padding_mask & causal_mask # (B, 1, S, S)
        mask = mask.expand(-1, self.transformer_blocks[0].attention.h, -1, -1) # (B, h, S, S)

        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x