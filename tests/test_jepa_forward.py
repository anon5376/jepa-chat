import torch

from jepa_chat.config import MaskConfig, ModelConfig
from jepa_chat.data.masking import sample_masks
from jepa_chat.models.jepa import TextJEPA


def _small_cfg() -> ModelConfig:
    return ModelConfig(max_len=32, d_model=32, n_heads=4, n_layers=1, predictor_d_model=16)


def test_forward_backward_and_ema():
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = TextJEPA(cfg)
    ids = torch.randint(0, 256, (2, 32))
    attn = torch.ones_like(ids, dtype=torch.bool)
    masks = sample_masks(attn, MaskConfig(), torch.Generator().manual_seed(0))

    out = model(ids, attn, masks.context_mask, masks.target_mask)
    assert torch.isfinite(out.loss)
    assert out.pred.shape == out.target.shape
    assert out.pred.shape[0] == masks.target_mask.sum().item()
    out.loss.backward()

    before = model.target_encoder.tok_emb.weight.clone()
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    model.update_target_encoder(0.5)
    assert not torch.allclose(before, model.target_encoder.tok_emb.weight)
    assert all(not p.requires_grad for p in model.target_encoder.parameters())


def test_embed_shape():
    cfg = _small_cfg()
    model = TextJEPA(cfg).eval()
    ids = torch.randint(0, 256, (3, 32))
    attn = torch.ones_like(ids, dtype=torch.bool)
    assert model.embed(ids, attn).shape == (3, cfg.d_model)
