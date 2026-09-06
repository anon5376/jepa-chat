"""Level 2: ConversationPredictor.

Operates on sequences of *turn latents* produced by the Level 1 TextJEPA. A causal
transformer builds a conversation state h_k from turns <= k and predicts the
(normalised) latent of turn k+1 given the role of that next turn. Level 1 latents
are detached, so this level cannot collapse by moving its own targets.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from jepa_chat.config import ConversationConfig


def normalize_latent(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Affine-free LayerNorm over the feature dim: removes scale games from the target."""
    return F.layer_norm(z, z.shape[-1:], eps=eps)


class ConversationPredictor(nn.Module):
    def __init__(self, cfg: ConversationConfig, latent_dim: int):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim
        d = cfg.d_model
        self.in_proj = nn.Linear(latent_dim, d)
        self.role_emb = nn.Embedding(cfg.n_roles, d)
        self.pos_emb = nn.Embedding(cfg.max_turns, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=4 * d,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, latent_dim))

    # ------------------------------------------------------------------ core
    def encode(self, z: torch.Tensor, roles: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """z (B,K,D_lat) turn latents; roles (B,K) long; attn (B,K) bool. Returns h (B,K,d)."""
        _, k, _ = z.shape
        pos = torch.arange(k, device=z.device)
        x = self.in_proj(normalize_latent(z)) + self.role_emb(roles) + self.pos_emb(pos)[None]
        causal = torch.triu(torch.ones(k, k, dtype=torch.bool, device=z.device), diagonal=1)
        h = self.blocks(x, mask=causal, src_key_padding_mask=~attn)
        return self.norm(h)

    def predict_from_state(self, h: torch.Tensor, next_roles: torch.Tensor) -> torch.Tensor:
        """h (..., d), next_roles (...) long -> predicted normalised latent (..., D_lat)."""
        return self.head(torch.cat([h, self.role_emb(next_roles)], dim=-1))

    def forward(
        self,
        z: torch.Tensor,
        roles: torch.Tensor,
        next_roles: torch.Tensor,
        attn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(z, roles, attn)
        return h, self.predict_from_state(h, next_roles)

    def loss(self, z: torch.Tensor, roles: torch.Tensor, attn: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Teacher-forced next-turn latent prediction loss over a padded batch."""
        target = normalize_latent(z[:, 1:])
        valid = attn[:, 1:] & attn[:, :-1]
        _, pred = self.forward(z[:, :-1], roles[:, :-1], roles[:, 1:], attn[:, :-1])
        pred = pred[valid]
        tgt = target[valid]
        loss = F.smooth_l1_loss(pred, tgt)
        with torch.no_grad():
            # chance baseline: same predictions vs shuffled targets
            perm = torch.randperm(tgt.shape[0], device=tgt.device)
            base = F.smooth_l1_loss(pred, tgt[perm])
            ratio = (loss / base.clamp_min(1e-8)).item()
        return loss, {"loss2": loss.item(), "chance": base.item(), "ratio": ratio}

    # -------------------------------------------------- planning hooks (later)
    @torch.no_grad()
    def rollout(
        self, z: torch.Tensor, roles: torch.Tensor, attn: torch.Tensor, future_roles: list[int]
    ) -> torch.Tensor:
        """Imagine len(future_roles) turns ahead by feeding predictions back. Returns (B,n,D)."""
        preds = []
        z_cur, r_cur, a_cur = z, roles, attn
        for role in future_roles:
            h = self.encode(z_cur, r_cur, a_cur)
            last = a_cur.sum(1) - 1
            h_last = h[torch.arange(h.shape[0]), last]
            nr = torch.full((h.shape[0],), role, dtype=torch.long, device=z.device)
            p = self.predict_from_state(h_last, nr)
            preds.append(p)
            z_cur = torch.cat([z_cur, p[:, None]], 1)[:, -self.cfg.max_turns :]
            r_cur = torch.cat([r_cur, nr[:, None]], 1)[:, -self.cfg.max_turns :]
            a_cur = torch.cat([a_cur, torch.ones_like(nr, dtype=torch.bool)[:, None]], 1)[:, -self.cfg.max_turns :]
        return torch.stack(preds, 1)

    @staticmethod
    @torch.no_grad()
    def score_candidates(pred: torch.Tensor, cand_z: torch.Tensor) -> torch.Tensor:
        """Negative L2 between predicted latent (D,) and candidate latents (N,D) -> (N,)."""
        return -((normalize_latent(cand_z) - pred[None]) ** 2).sum(-1)
