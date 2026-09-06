import torch

from jepa_chat.config import Config, ConversationConfig, ModelConfig
from jepa_chat.core.checkpoint import build_models, load_checkpoint, save_checkpoint
from jepa_chat.inference.chat import load_model


def _cfg():
    return Config(
        model=ModelConfig(max_len=32, d_model=32, n_heads=4, n_layers=1, predictor_d_model=16),
        conversation=ConversationConfig(d_model=32, n_layers=1, max_turns=8),
    )


def test_v2_roundtrip(tmp_path):
    cfg = _cfg()
    l1, l2 = build_models(cfg)
    opt1 = torch.optim.AdamW(l1.encoder.parameters())
    p = save_checkpoint(tmp_path / "c.pt", cfg, l1, l2, opt1=opt1, step=7, enc_version=3, pretrain_latent_std=0.9)
    got = load_checkpoint(p)
    assert got.step == 7 and got.enc_version == 3 and got.format_version == 2
    assert got.pretrain_latent_std == 0.9 and got.opt1_state is not None
    for a, b in zip(l2.parameters(), got.level2.parameters(), strict=True):
        assert torch.equal(a, b)
    assert got.cfg.model.max_len == 32


def test_v1_upgrade(tmp_path):
    cfg = _cfg()
    l1, _ = build_models(cfg)
    v1 = {"config": cfg.to_dict(), "model": l1.state_dict()}
    p = tmp_path / "old.pt"
    torch.save(v1, p)
    got = load_checkpoint(p)
    assert got.format_version == 1 and got.step == 0
    assert got.level2 is not None
    assert load_model(p) is not None
    assert load_model(tmp_path / "missing.pt") is None
