"""Transformer text encoder used for both the online (context) and EMA (target) branches."""

from __future__ import annotations

import torch
from torch import nn

from jepa_chat.config import ModelConfig


class TextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, ids: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        """Encode a batch of token ids.

        ids:     (B, T) long
        visible: (B, T) bool, True at positions the encoder may attend to.
        Returns (B, T, D). Outputs at non-visible positions are meaningless.
        """
        _, seq_len = ids.shape
        pos = torch.arange(seq_len, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None]
        x = self.blocks(x, src_key_padding_mask=~visible)
        return self.norm(x)
