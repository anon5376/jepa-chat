"""Latent -> text decoding.

A JEPA predicts *representations*, not tokens. To talk back to a user we need a
decoder. The long-term plan is an LLM conditioned on JEPA latents; the class here
is a deliberately crude placeholder so the pipeline runs end to end today.
"""

from __future__ import annotations

from typing import Protocol

import torch

from jepa_chat.data.tokenizer import ByteTokenizer
from jepa_chat.models.jepa import TextJEPA


class LatentDecoder(Protocol):
    def decode(self, latents: torch.Tensor) -> str: ...


class NearestTokenDecoder:
    """Maps each latent to the nearest token in the target encoder's *input* embedding table.

    This ignores context and is only a smoke-test decoder. It will produce noisy bytes.
    """

    def __init__(self, model: TextJEPA, tokenizer: ByteTokenizer | None = None):
        self.model = model
        self.tokenizer = tokenizer or ByteTokenizer()
        table = model.target_encoder.tok_emb.weight.detach()  # (V, D)
        self.table = torch.nn.functional.normalize(table, dim=-1)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> str:
        z = torch.nn.functional.normalize(latents.to(self.table.device), dim=-1)  # (N, D)
        ids = (z @ self.table.T).argmax(dim=-1)
        return self.tokenizer.decode(ids)
