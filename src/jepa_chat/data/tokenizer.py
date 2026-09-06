"""A dependency-free byte-level tokenizer.

Bytes 0-255 map to themselves; two special ids are appended for PAD and MASK.
This keeps the toy experiments self-contained. Swap in a BPE tokenizer later
once the architecture is stable.
"""

from __future__ import annotations

import torch


class ByteTokenizer:
    PAD = 256
    MASK = 257
    vocab_size = 258

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        ids = list(text.encode("utf-8", errors="replace"))
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        raw = bytes(i for i in ids if 0 <= i < 256)
        return raw.decode("utf-8", errors="replace")

    def batch_encode(self, texts: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (ids, attention_mask) padded to max_len. attention_mask is 1 for real tokens."""
        ids = torch.full((len(texts), max_len), self.PAD, dtype=torch.long)
        attn = torch.zeros((len(texts), max_len), dtype=torch.bool)
        for row, text in enumerate(texts):
            enc = self.encode(text, max_len)
            ids[row, : len(enc)] = torch.tensor(enc, dtype=torch.long)
            attn[row, : len(enc)] = True
        return ids, attn
