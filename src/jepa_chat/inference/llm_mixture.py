"""Mixture-of-LLMs routing.

The JEPA provides a latent summary of the conversation. That latent can drive a
router that decides which backend (local model, hosted API, tool, specialist) should
produce the surface text. Backends are kept behind a tiny protocol so provider
adapters can be added without touching the rest of the system.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch


class LLMBackend(Protocol):
    name: str

    def generate(self, prompt: str, history: Sequence[tuple[str, str]]) -> str: ...


@dataclass
class EchoBackend:
    """Offline stand-in so the chat loop works without any model or API key."""

    name: str = "echo"

    def generate(self, prompt: str, history: Sequence[tuple[str, str]]) -> str:
        return f"[{self.name}] You said: {prompt}"


RouteFn = Callable[[torch.Tensor | None, str], int]


@dataclass
class MixtureRouter:
    """Chooses one backend per turn.

    route_fn receives (latent, prompt) and returns a backend index. The default
    round-robins, which is only useful as a placeholder; a learned router on top
    of the JEPA latent is the intended replacement.
    """

    backends: list[LLMBackend]
    route_fn: RouteFn | None = None
    _turn: int = field(default=0, init=False)

    def route(self, latent: torch.Tensor | None, prompt: str) -> LLMBackend:
        if not self.backends:
            raise ValueError("MixtureRouter has no backends")
        if self.route_fn is not None:
            idx = self.route_fn(latent, prompt)
        else:
            idx = self._turn % len(self.backends)
        self._turn += 1
        return self.backends[idx % len(self.backends)]

    def generate(
        self, latent: torch.Tensor | None, prompt: str, history: Sequence[tuple[str, str]]
    ) -> tuple[str, str]:
        backend = self.route(latent, prompt)
        return backend.name, backend.generate(prompt, history)
