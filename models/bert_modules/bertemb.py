from torch import nn as nn
import torch

from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class BERTEmbeddingTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)

        max_len = args.bert_max_len
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # Create attention mask based on non-zero vectors (non-padded)
        # paddingが0ベクトルなので、絶対値の和が0より大きいかどうかで判定
        # language modelのembeddingが0ベクトルになると衝突するが、そんなことないでしょ、、、
        mask = (torch.abs(x).sum(dim=-1) > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass