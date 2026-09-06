"""I-JEPA style span masking adapted to 1-D token sequences.

For each sequence we sample a few contiguous *target* spans. The context
encoder sees everything except the targets; the predictor must predict the
target latents from the context latents. Nothing is ever reconstructed in
token space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from jepa_chat.config import MaskConfig


@dataclass
class MaskSample:
    context_mask: torch.Tensor  # (B, T) bool, True where the context encoder may attend
    target_mask: torch.Tensor  # (B, T) bool, True at positions to predict


def sample_masks(
    attention_mask: torch.Tensor,
    cfg: MaskConfig,
    generator: torch.Generator | None = None,
) -> MaskSample:
    """Sample context/target masks for a padded batch.

    attention_mask: (B, T) bool, True for real (non-pad) tokens.
    """
    batch, seq_len = attention_mask.shape
    target = torch.zeros((batch, seq_len), dtype=torch.bool)
    lo, hi = cfg.target_span_frac

    for b in range(batch):
        length = int(attention_mask[b].sum().item())
        if length < 4:
            continue
        max_target_total = int(length * (1.0 - cfg.min_context_frac))
        placed = 0
        for _ in range(cfg.num_targets):
            span_frac = lo + (hi - lo) * torch.rand((), generator=generator).item()
            span = max(1, int(round(span_frac * length)))
            if placed + span > max_target_total:
                span = max_target_total - placed
                if span <= 0:
                    break
            start = int(torch.randint(0, length - span + 1, (), generator=generator).item())
            target[b, start : start + span] = True
            placed += span

    context = attention_mask & ~target
    return MaskSample(context_mask=context, target_mask=target)
