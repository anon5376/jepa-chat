"""TextJEPA: online encoder + EMA target encoder + latent predictor."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from jepa_chat.config import ModelConfig
from jepa_chat.models.encoder import TextEncoder
from jepa_chat.models.predictor import LatentPredictor


@dataclass
class JEPAOutput:
    loss: torch.Tensor
    pred: torch.Tensor  # (N, D) predicted target latents
    target: torch.Tensor  # (N, D) EMA target latents
    latent_std: torch.Tensor  # scalar: mean per-dim std of targets, a cheap collapse monitor


class TextJEPA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TextEncoder(cfg)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = LatentPredictor(cfg)

    def forward(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> JEPAOutput:
        with torch.no_grad():
            full = self.target_encoder(ids, attention_mask)
            target = full[target_mask]

        ctx = self.encoder(ids, context_mask)
        pred_all = self.predictor(ctx, context_mask, target_mask)
        pred = pred_all[target_mask]

        loss = F.smooth_l1_loss(pred, target)
        latent_std = target.std(dim=0).mean() if target.shape[0] > 1 else target.new_zeros(())
        return JEPAOutput(loss=loss, pred=pred, target=target, latent_std=latent_std)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters(), strict=True):
            p_target.mul_(momentum).add_(p_online.detach(), alpha=1.0 - momentum)

    @torch.no_grad()
    def embed(self, ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pooled target-encoder embedding of a full sequence: (B, D)."""
        latents = self.target_encoder(ids, attention_mask)
        weights = attention_mask.float()[..., None]
        return (latents * weights).sum(1) / weights.sum(1).clamp_min(1.0)
