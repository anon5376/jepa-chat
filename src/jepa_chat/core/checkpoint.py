"""Checkpoint v2: both levels + optimizers + step + encoder version. Loads v1 files too."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from jepa_chat.config import Config
from jepa_chat.models.conversation import ConversationPredictor
from jepa_chat.models.jepa import TextJEPA

FORMAT_VERSION = 2


@dataclass
class Loaded:
    cfg: Config
    level1: TextJEPA
    level2: ConversationPredictor
    opt1_state: dict | None
    opt2_state: dict | None
    step: int
    enc_version: int
    format_version: int
    pretrain_latent_std: float | None


def build_models(cfg: Config) -> tuple[TextJEPA, ConversationPredictor]:
    level1 = TextJEPA(cfg.model)
    level2 = ConversationPredictor(cfg.conversation, latent_dim=cfg.model.d_model)
    return level1, level2


def save_checkpoint(
    path: str | Path,
    cfg: Config,
    level1: TextJEPA,
    level2: ConversationPredictor,
    opt1: torch.optim.Optimizer | None = None,
    opt2: torch.optim.Optimizer | None = None,
    step: int = 0,
    enc_version: int = 0,
    pretrain_latent_std: float | None = None,
) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "config": cfg.to_dict(),
        "step": step,
        "enc_version": enc_version,
        "pretrain_latent_std": pretrain_latent_std,
        "level1": {k: v.detach().cpu() for k, v in level1.state_dict().items()},
        "level2": {k: v.detach().cpu() for k, v in level2.state_dict().items()},
        "opt1": opt1.state_dict() if opt1 is not None else None,
        "opt2": opt2.state_dict() if opt2 is not None else None,
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".ckpt-", suffix=".tmp")
    os.close(fd)
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Loaded:
    ckpt = torch.load(Path(path).expanduser(), map_location=map_location, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    level1, level2 = build_models(cfg)
    if "format_version" not in ckpt:  # v1: {"config", "model"}
        level1.load_state_dict(ckpt["model"])
        return Loaded(cfg, level1, level2, None, None, 0, 0, 1, None)
    level1.load_state_dict(ckpt["level1"])
    level2.load_state_dict(ckpt["level2"])
    return Loaded(
        cfg=cfg,
        level1=level1,
        level2=level2,
        opt1_state=ckpt.get("opt1"),
        opt2_state=ckpt.get("opt2"),
        step=int(ckpt.get("step", 0)),
        enc_version=int(ckpt.get("enc_version", 0)),
        format_version=int(ckpt["format_version"]),
        pretrain_latent_std=ckpt.get("pretrain_latent_std"),
    )
