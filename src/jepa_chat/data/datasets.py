"""Tiny text corpus utilities for toy experiments."""

from __future__ import annotations

from pathlib import Path

import torch

from jepa_chat.data.tokenizer import ByteTokenizer

_SAMPLE_TEXT = """
A joint-embedding predictive architecture learns to predict the representation of
missing parts of an input from the representation of the visible parts. It never
reconstructs pixels or tokens directly. Instead, an online encoder embeds the context,
a slowly moving target encoder embeds the whole input, and a small predictor learns to
map from context embeddings to target embeddings. Because prediction happens in latent
space, the model can ignore unpredictable surface details and focus on the structure
that actually matters for the task. This is the idea we want to push toward a chat
assistant: a latent world model of the conversation, and language models used only as
decoders, tools, or specialist experts on top of it.
"""


def load_corpus(path: str | Path | None) -> str:
    if path is None:
        return _SAMPLE_TEXT.strip()
    return Path(path).read_text(encoding="utf-8")


class ChunkedTextDataset:
    """Cuts a corpus into random fixed-length byte windows."""

    def __init__(self, text: str, seq_len: int, tokenizer: ByteTokenizer | None = None):
        self.tokenizer = tokenizer or ByteTokenizer()
        self.ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        self.seq_len = seq_len
        if len(self.ids) < seq_len:
            reps = seq_len // len(self.ids) + 1
            self.ids = self.ids.repeat(reps)

    def sample_batch(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.ids) - self.seq_len
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
        ids = torch.stack([self.ids[s : s + self.seq_len] for s in starts])
        attn = torch.ones_like(ids, dtype=torch.bool)
        return ids, attn
