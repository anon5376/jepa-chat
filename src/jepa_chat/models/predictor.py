"""Latent predictor: context latents + mask tokens -> predicted target latents."""

from __future__ import annotations

import torch
from torch import nn

from jepa_chat.config import ModelConfig


class LatentPredictor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.inp = nn.Linear(cfg.d_model, cfg.predictor_d_model)
        self.mask_token = nn.Parameter(torch.zeros(cfg.predictor_d_model))
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.predictor_d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.predictor_d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.predictor_d_model,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.predictor_n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.predictor_d_model)
        self.out = nn.Linear(cfg.predictor_d_model, cfg.d_model)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context_latents: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        context_latents: (B, T, D) from the online encoder
        context_mask:    (B, T) bool, True where context is valid
        target_mask:     (B, T) bool, True where we must predict
        Returns (B, T, D); read off values at target positions.
        """
        _, seq_len, _ = context_latents.shape
        pos = self.pos_emb(torch.arange(seq_len, device=context_latents.device))[None]
        x = self.inp(context_latents)
        x = torch.where(target_mask[..., None], self.mask_token.expand_as(x), x)
        x = x + pos
        visible = context_mask | target_mask
        x = self.blocks(x, src_key_padding_mask=~visible)
        return self.out(self.norm(x))
