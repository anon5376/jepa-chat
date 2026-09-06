import numpy as np
import pytest
import torch

from jepa_chat.config import Config, ConversationConfig, CoreConfig, ModelConfig
from jepa_chat.core.checkpoint import build_models
from jepa_chat.core.engine import ModelThread
from jepa_chat.core.memory import MemoryStore
from jepa_chat.core.service import CoreService
from jepa_chat.core.state import CoreState, EventBus


def tiny_config(home: str) -> Config:
    return Config(
        model=ModelConfig(max_len=64, d_model=32, n_heads=4, n_layers=1, predictor_d_model=16),
        conversation=ConversationConfig(d_model=32, n_layers=1, max_turns=8, head_bytes=48, tail_bytes=16),
        core=CoreConfig(
            home=home,
            checkpoint=f"{home}/core.pt",
            online_batch_sessions=2,
            online_window=4,
            snapshot_every=2,
            checkpoint_every=4,
            reembed_every=3,
            map_points=50,
        ),
    )


@pytest.fixture
def runtime(tmp_path):
    torch.manual_seed(0)
    cfg = tiny_config(str(tmp_path))
    cfg.train.seq_len = 64
    l1, l2 = build_models(cfg)
    bus = EventBus()
    state = CoreState()
    memory = MemoryStore(tmp_path / "mem.sqlite", cfg.model.d_model)
    engine = ModelThread(
        cfg,
        l1,
        l2,
        memory,
        bus,
        state,
        torch.device("cpu"),
        checkpoint_path=str(tmp_path / "core.pt"),
        pretrain_latent_std=1.0,
        train=False,
    )
    svc = CoreService(cfg, memory, engine, bus, state)
    yield cfg, memory, engine, bus, state, svc
    engine.stop()
    if engine.is_alive():
        engine.join(10)
    memory.close()


def rand_latent(dim: int = 32, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
