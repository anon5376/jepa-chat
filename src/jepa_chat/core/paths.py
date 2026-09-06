"""Filesystem layout under JEPA_HOME (default ~/.jepa-chat)."""

from __future__ import annotations

import os
from pathlib import Path


def jepa_home() -> Path:
    return Path(os.environ.get("JEPA_HOME", "~/.jepa-chat")).expanduser()


def ensure_dirs(home: Path | None = None) -> Path:
    home = home or jepa_home()
    for sub in ("data", "checkpoints", "hooks", "logs"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    return home


def default_checkpoint(home: Path | None = None) -> Path:
    return (home or jepa_home()) / "checkpoints" / "core.pt"


def memory_db(home: Path | None = None) -> Path:
    return (home or jepa_home()) / "memory.sqlite"


def turns_jsonl(home: Path | None = None) -> Path:
    return (home or jepa_home()) / "data" / "turns.jsonl"
