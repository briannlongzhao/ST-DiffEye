import math

import torch
from torch import nn


class PositionEmbeddingSine2d(nn.Module):
    """Fixed 2D sine/cosine positional encoding added to flattened image patch features."""

    def __init__(self, spatial_dim, hidden_dim=768, temperature=10000):
        super().__init__()
        num_pos_feats = hidden_dim // 2
        h, w = spatial_dim
        position_y = torch.arange(h).unsqueeze(1)
        position_x = torch.arange(w).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_pos_feats, 2) * (-math.log(temperature) / num_pos_feats))
        pe_y = torch.zeros(h, 1, num_pos_feats)
        pe_x = torch.zeros(1, w, num_pos_feats)
        pe_y[:, 0, 0::2] = torch.sin(position_y * div_term)
        pe_y[:, 0, 1::2] = torch.cos(position_y * div_term)
        pe_x[0, :, 0::2] = torch.sin(position_x * div_term)
        pe_x[0, :, 1::2] = torch.cos(position_x * div_term)
        pos = torch.cat((pe_y.repeat(1, w, 1), pe_x.repeat(h, 1, 1)), dim=-1).permute(2, 0, 1)
        self.register_buffer("pos", pos.reshape(hidden_dim, -1).permute(1, 0).unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pos
