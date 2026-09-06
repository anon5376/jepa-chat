"""Datasets over sessions of turns, for Level 1 (bytes) and Level 2 (turn latents)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from jepa_chat.data.datasets import ChunkedTextDataset
from jepa_chat.data.tokenizer import ByteTokenizer
from jepa_chat.data.turns import Session, truncate_turn
from jepa_chat.models.jepa import TextJEPA


def sessions_to_text(sessions: Sequence[Session]) -> str:
    """Flatten sessions to one text stream for byte-level Level 1 pretraining."""
    return "\n\n".join("\n".join(f"{t.role}: {t.text}" for t in s.turns) for s in sessions)


def turn_text_dataset(sessions: Sequence[Session], seq_len: int) -> ChunkedTextDataset:
    return ChunkedTextDataset(sessions_to_text(sessions), seq_len)


@torch.no_grad()
def embed_texts(
    model: TextJEPA,
    texts: Sequence[str],
    tokenizer: ByteTokenizer,
    max_len: int,
    head: int,
    tail: int,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Turn texts -> (N, D) float32 latents from the target encoder."""
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = [truncate_turn(t, head, tail) for t in texts[i : i + batch_size]]
        ids, attn = tokenizer.batch_encode(chunk, max_len)
        z = model.embed(ids.to(device), attn.to(device))
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, model.cfg.d_model), np.float32)


class LatentCache:
    """Per-session turn latents + roles, the training set for Level 2."""

    def __init__(self, latents: list[np.ndarray], roles: list[np.ndarray]):
        assert len(latents) == len(roles)
        self.latents = latents
        self.roles = roles
        self.lengths = np.array([len(r) for r in roles])

    @classmethod
    def build(
        cls,
        model: TextJEPA,
        sessions: Sequence[Session],
        tokenizer: ByteTokenizer,
        max_len: int,
        head: int,
        tail: int,
        device: torch.device | str = "cpu",
    ) -> LatentCache:
        texts = [t.text for s in sessions for t in s.turns]
        all_z = embed_texts(model, texts, tokenizer, max_len, head, tail, device)
        latents, roles, i = [], [], 0
        for s in sessions:
            n = len(s.turns)
            latents.append(all_z[i : i + n])
            roles.append(np.array([t.role_id for t in s.turns], dtype=np.int64))
            i += n
        return cls(latents, roles)

    def sample_batch(
        self, batch_size: int, max_turns: int, rng: np.random.Generator, min_turns: int = 2
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random windows of <= max_turns turns, left-aligned, padded. Returns (z, roles, attn)."""
        eligible = np.flatnonzero(self.lengths >= min_turns)
        if len(eligible) == 0:
            raise ValueError("no session has enough turns")
        dim = self.latents[eligible[0]].shape[1]
        z = torch.zeros((batch_size, max_turns, dim))
        roles = torch.full((batch_size, max_turns), 2, dtype=torch.long)
        attn = torch.zeros((batch_size, max_turns), dtype=torch.bool)
        for b in range(batch_size):
            s = int(rng.choice(eligible))
            n = int(self.lengths[s])
            win = min(n, max_turns)
            start = int(rng.integers(0, n - win + 1))
            z[b, :win] = torch.from_numpy(self.latents[s][start : start + win])
            roles[b, :win] = torch.from_numpy(self.roles[s][start : start + win])
            attn[b, :win] = True
        return z, roles, attn

    def split(self, holdout: float, rng: np.random.Generator) -> tuple[LatentCache, LatentCache]:
        idx = rng.permutation(len(self.latents))
        n_hold = max(1, int(len(idx) * holdout)) if len(idx) > 1 else 0
        hold, train = idx[:n_hold], idx[n_hold:]
        pick = lambda ids: LatentCache([self.latents[i] for i in ids], [self.roles[i] for i in ids])  # noqa: E731
        return pick(train), pick(hold)
