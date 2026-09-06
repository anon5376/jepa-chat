import torch

from jepa_chat.config import MaskConfig
from jepa_chat.data.masking import sample_masks


def test_masks_are_disjoint_and_within_attention():
    attn = torch.ones((4, 64), dtype=torch.bool)
    attn[1, 40:] = False  # a padded row
    masks = sample_masks(attn, MaskConfig(), torch.Generator().manual_seed(0))
    assert masks.context_mask.shape == attn.shape
    assert not (masks.context_mask & masks.target_mask).any()
    assert not (masks.target_mask & ~attn).any()
    assert masks.target_mask.any()


def test_min_context_is_respected():
    cfg = MaskConfig(num_targets=4, target_span_frac=(0.3, 0.3), min_context_frac=0.5)
    attn = torch.ones((8, 100), dtype=torch.bool)
    masks = sample_masks(attn, cfg, torch.Generator().manual_seed(1))
    assert (masks.context_mask.sum(1) >= 50).all()
