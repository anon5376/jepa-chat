import torch

from jepa_chat.config import ConversationConfig
from jepa_chat.models.conversation import ConversationPredictor, normalize_latent


def _model():
    torch.manual_seed(0)
    return ConversationPredictor(ConversationConfig(d_model=32, n_heads=4, n_layers=2, max_turns=8), 16).eval()


def test_shapes_and_loss():
    m = _model()
    z = torch.randn(3, 8, 16)
    roles = torch.randint(0, 2, (3, 8))
    attn = torch.ones(3, 8, dtype=torch.bool)
    attn[1, 5:] = False
    h, pred = m(z, roles, roles, attn)
    assert h.shape == (3, 8, 32) and pred.shape == (3, 8, 16)
    loss, stats = m.loss(z, roles, attn)
    assert torch.isfinite(loss) and "ratio" in stats


def test_causality():
    m = _model()
    z = torch.randn(1, 8, 16)
    roles = torch.zeros(1, 8, dtype=torch.long)
    attn = torch.ones(1, 8, dtype=torch.bool)
    _, p1 = m(z, roles, roles, attn)
    z2 = z.clone()
    z2[0, 6:] += torch.randn(2, 16) * 5.0  # perturb the future
    _, p2 = m(z2, roles, roles, attn)
    assert torch.allclose(p1[0, :6], p2[0, :6], atol=1e-5)
    assert not torch.allclose(p1[0, 6:], p2[0, 6:])


def test_rollout_and_scoring():
    m = _model()
    z = torch.randn(2, 4, 16)
    roles = torch.zeros(2, 4, dtype=torch.long)
    attn = torch.ones(2, 4, dtype=torch.bool)
    fut = m.rollout(z, roles, attn, [1, 0, 1])
    assert fut.shape == (2, 3, 16)
    scores = ConversationPredictor.score_candidates(fut[0, 0], torch.randn(5, 16))
    assert scores.shape == (5,)
    n = normalize_latent(torch.randn(4, 16))
    assert torch.allclose(n.mean(-1), torch.zeros(4), atol=1e-5)
