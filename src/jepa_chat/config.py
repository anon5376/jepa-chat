"""Configuration dataclasses and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 258  # 256 bytes + PAD + MASK
    max_len: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    predictor_d_model: int = 64
    predictor_n_layers: int = 2
    dropout: float = 0.0
    var_coef: float = 0.0  # VICReg-style variance term on online-encoder latents (anti-collapse)
    cov_coef: float = 0.0  # VICReg-style covariance term (anti dimensional collapse)


@dataclass
class MaskConfig:
    num_targets: int = 2  # number of target spans per sequence
    target_span_frac: tuple[float, float] = (0.10, 0.25)  # span length as fraction of seq len
    min_context_frac: float = 0.4  # guarantee at least this much context remains


@dataclass
class TrainConfig:
    corpus_path: str | None = None  # plain-text file; falls back to a built-in sample
    seq_len: int = 128
    batch_size: int = 32
    steps: int = 500
    lr: float = 3e-4
    weight_decay: float = 0.05
    ema_start: float = 0.996
    ema_end: float = 1.0
    log_every: int = 25
    checkpoint_path: str = "checkpoints/text_jepa.pt"
    seed: int = 0
    device: str = "auto"  # auto | cpu | mps | cuda


@dataclass
class ConversationConfig:
    """Level 2: predictor over sequences of turn latents."""

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_turns: int = 64
    n_roles: int = 3  # 0 = user, 1 = assistant, 2 = other/system
    dropout: float = 0.1
    head_bytes: int = 384  # turn truncation: keep this many bytes from the start
    tail_bytes: int = 128  # and this many from the end


@dataclass
class CoreConfig:
    """Long-running daemon settings."""

    host: str = "127.0.0.1"
    port: int = 8765
    home: str = "~/.jepa-chat"
    checkpoint: str = "~/.jepa-chat/checkpoints/core.pt"
    device: str = "auto"
    online_lr: float = 1e-4
    online_batch_sessions: int = 16
    online_window: int = 32
    live_ratio: float = 0.25
    ema_momentum: float = 0.998
    snapshot_every: int = 25
    checkpoint_every: int = 300
    reembed_every: int = 500
    train_on_start: bool = True
    recall_k: int = 5
    map_points: int = 200
    collapse_floor: float = 0.3  # pause training if latent_std < floor * pretrain value
    steps_per_new_turn: int = 20  # online steps earned per newly observed live turn
    idle_steps_per_min: float = 2.0  # slow background replay rate when nothing new arrives
    max_step_budget: int = 400  # cap on banked steps so a burst of turns cannot spin for hours


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    core: CoreConfig = field(default_factory=CoreConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        mask_raw = dict(raw.get("mask", {}))
        if "target_span_frac" in mask_raw:
            mask_raw["target_span_frac"] = tuple(mask_raw["target_span_frac"])
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            mask=MaskConfig(**mask_raw),
            train=TrainConfig(**raw.get("train", {})),
            conversation=ConversationConfig(**raw.get("conversation", {})),
            core=CoreConfig(**raw.get("core", {})),
        )

    @classmethod
    def from_dict(cls, raw: dict) -> Config:
        mask_raw = dict(raw.get("mask", {}))
        if "target_span_frac" in mask_raw:
            mask_raw["target_span_frac"] = tuple(mask_raw["target_span_frac"])
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            mask=MaskConfig(**mask_raw),
            train=TrainConfig(**raw.get("train", {})),
            conversation=ConversationConfig(**raw.get("conversation", {})),
            core=CoreConfig(**raw.get("core", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)
