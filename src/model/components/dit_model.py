# DiT backbone adapted from:
#   DiT (https://github.com/facebookresearch/DiT), GLIDE (https://github.com/openai/glide-text2im)
#   and ScanDiff (https://github.com/aimagelab/ScanDiff).

import math

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

from src.model.components.autoencoder import Decoder, Encoder
from src.model.components.img_text_projector import ImgTextProjector
from src.utils.positional_encodings import PositionEmbeddingSine2d

FIXATION_STRATEGIES = ("data_channel", "latent_space")


class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class DiTBlock(nn.Module):
    """Transformer block: self-attention over gaze tokens, cross-attention to the condition, MLP."""

    def __init__(self, hidden_size, num_heads, dim_feedforward=2048):
        super().__init__()
        self.attn = MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_attn = MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, x, c):
        out = x
        q = k = v = out
        out = self.norm1(out + self.dropout1(self.attn(q, k, v)[0]))
        # Queries are the pre-attention tokens; keys/values come from the condition.
        out = self.norm2(out + self.dropout2(self.cross_attn(q, c, c)[0]))
        out2 = self.linear2(self.dropout(self.activation(self.linear1(out))))
        return self.norm3(out + self.dropout3(out2))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)

    def forward(self, x):
        return self.linear(self.norm_final(x))


class DiTModel(nn.Module):
    """ST-DiffEye denoiser.

    Each gaze token is a trajectory sample (x, y, dt) augmented with its normalized
    fixation index. ``fixation_strategy`` selects where the index is concatenated:
      - ``data_channel``: as a 4th raw input channel before the encoder (default).
      - ``latent_space``: as an extra channel of the encoded latent.
    """

    def __init__(
        self,
        encoder,
        decoder,
        max_len=750,
        spatial_dim=(37, 37),
        scanpath_emb_size=256,
        hidden_size=512,
        img_feature_dim=768,
        text_feature_dim=512,
        depth=6,
        num_heads=8,
        fixation_max_len=16,
        fixation_strategy="data_channel",
    ):
        super().__init__()
        fixation_strategy = fixation_strategy.lower()
        if fixation_strategy not in FIXATION_STRATEGIES:
            raise ValueError(
                f"Unsupported fixation_strategy='{fixation_strategy}'. Choose from {FIXATION_STRATEGIES}."
            )
        self.max_len = max_len
        self.scanpath_emb_size = scanpath_emb_size
        self.fixation_max_len = fixation_max_len
        self.fixation_strategy = fixation_strategy

        if fixation_strategy == "data_channel":
            # The fixation index is one extra input/output channel of the autoencoder.
            encoder_dims = self._linear_stack_dims(encoder)
            decoder_dims = self._linear_stack_dims(decoder)
            encoder = Encoder([encoder_dims[0] + 1] + encoder_dims[1:])
            decoder = Decoder(decoder_dims[:-1] + [decoder_dims[-1] + 1])
            self.diffusion_latent_dim = scanpath_emb_size
        else:
            # The fixation index is one extra channel of the diffusion latent.
            self.diffusion_latent_dim = scanpath_emb_size + 1
        self.encoder = encoder
        self.decoder = decoder

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = nn.Embedding(max_len, hidden_size)
        self.img_proj = nn.Linear(img_feature_dim, hidden_size)
        self.img_patch_pos = PositionEmbeddingSine2d(spatial_dim, hidden_size)
        self.img_text_projector = ImgTextProjector(img_feature_dim, text_feature_dim, hidden_size)
        self.input_up_proj = nn.Linear(self.diffusion_latent_dim, hidden_size)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, hidden_size)
        self.output_down_proj = nn.Linear(hidden_size, self.diffusion_latent_dim)
        self.token_validity_predictor = nn.Linear(self.diffusion_latent_dim, 2)
        self.register_buffer("position_ids", torch.arange(max_len), persistent=False)

    def forward(self, x, t, y, task_embedding, **kwargs):
        """
        x: (N, max_len, diffusion_latent_dim) noisy gaze latents
        t: (N,) diffusion timesteps
        y: (N, num_patches, img_feature_dim) image features
        task_embedding: (N, text_feature_dim) task text embedding
        """
        x = self.input_up_proj(x) + self.t_embedder(t).unsqueeze(1) + self.pos_embed(self.position_ids)
        c = self.img_text_projector(self.img_patch_pos(self.img_proj(y)), task_embedding)
        for block in self.blocks:
            x = block(x, c)
        return self.output_down_proj(self.final_layer(x))

    def get_embeds(self, trajectory, fixation_indices):
        """Encode ground-truth trajectories (N, T, 3) and fixation indices (N, T) into diffusion latents."""
        fixation_channel = self._normalize_fixation_indices(fixation_indices.to(trajectory.device))
        if self.fixation_strategy == "data_channel":
            return self.encoder(torch.cat([trajectory, fixation_channel], dim=-1))
        return torch.cat([self.encoder(trajectory), fixation_channel], dim=-1)

    def build_data_channel_scanpath(self, trajectory, fixation_indices):
        """Ground-truth reconstruction target for data_channel: (x, y, dt, fixation index)."""
        fixation_channel = self._normalize_fixation_indices(fixation_indices.to(trajectory.device))
        return torch.cat([trajectory, fixation_channel], dim=-1)

    def get_coords_and_time(self, model_output):
        """Decode latents to (x, y, dt[, fixation index]) per token."""
        if self.fixation_strategy == "latent_space":
            model_output = model_output[:, :, : self.scanpath_emb_size]
        return self.decoder(model_output)

    def _normalize_fixation_indices(self, fixation_indices):
        """Scale fixation indices to [0, 1]; negative (padding) values are kept as-is."""
        max_index = max(1.0, float(self.fixation_max_len + 1))
        normalized = fixation_indices.clone()
        valid = fixation_indices >= 0
        normalized[valid] = torch.clamp(fixation_indices[valid], min=0.0, max=max_index) / max_index
        return normalized.unsqueeze(-1)

    @staticmethod
    def _linear_stack_dims(module):
        dims = []
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                if not dims:
                    dims.append(int(layer.in_features))
                dims.append(int(layer.out_features))
        return dims
