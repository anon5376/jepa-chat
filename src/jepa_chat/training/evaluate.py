"""jepa-eval: is the core learning anything real? Held-out surprise test, retrieval sanity, probes."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jepa_chat.core.checkpoint import load_checkpoint
from jepa_chat.core.paths import default_checkpoint, turns_jsonl
from jepa_chat.data.conversation_dataset import LatentCache
from jepa_chat.data.tokenizer import ByteTokenizer
from jepa_chat.data.turns import read_jsonl
from jepa_chat.models.conversation import normalize_latent
from jepa_chat.training.train import pick_device


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) via rank statistic."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_s = np.concatenate([pos, neg])
    ranks = all_s.argsort().argsort().astype(np.float64) + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def surprise_test(level2, cache: LatentCache, max_turns: int, rng: np.random.Generator) -> dict:
    """For every held-out turn k>=1: error of pred_{k-1} vs true z_k, vs a random other turn."""
    device = next(level2.parameters()).device
    errs_true, errs_same, errs_other = [], [], []
    all_z = np.concatenate(cache.latents) if cache.latents else np.zeros((0, 1))
    for z_np, r_np in zip(cache.latents, cache.roles, strict=True):
        n = len(r_np)
        if n < 2:
            continue
        z = torch.from_numpy(z_np[-max_turns:])[None].to(device)
        r = torch.from_numpy(r_np[-max_turns:])[None].to(device)
        attn = torch.ones_like(r, dtype=torch.bool)
        _, pred = level2(z[:, :-1], r[:, :-1], r[:, 1:], attn[:, :-1])
        pred = pred[0].cpu().numpy()
        tgt = normalize_latent(z[0, 1:]).cpu().numpy()
        for k in range(pred.shape[0]):
            errs_true.append(((pred[k] - tgt[k]) ** 2).sum())
            j = int(rng.integers(0, tgt.shape[0]))
            errs_same.append(((pred[k] - tgt[j]) ** 2).sum())
            o = normalize_latent(torch.from_numpy(all_z[rng.integers(0, all_z.shape[0])][None]))[0].numpy()
            errs_other.append(((pred[k] - o) ** 2).sum())
    t, s, o = map(np.array, (errs_true, errs_same, errs_other))
    return {
        "n_pred": int(len(t)),
        "ratio_vs_same_session": float(t.mean() / s.mean()) if len(t) else float("nan"),
        "ratio_vs_other_session": float(t.mean() / o.mean()) if len(t) else float("nan"),
        "auroc_true_vs_same": auroc(-t, -s),
        "auroc_true_vs_other": auroc(-t, -o),
    }


def retrieval_sanity(cache: LatentCache) -> dict:
    """Nearest neighbour (excluding self) of each turn: same session rate vs chance."""
    z = np.concatenate(cache.latents)
    sid = np.concatenate([np.full(len(r), i) for i, r in enumerate(cache.roles)])
    mu, sd = z.mean(0), z.std(0) + 1e-5
    w = (z - mu) / sd
    w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
    sims = w @ w.T
    np.fill_diagonal(sims, -np.inf)
    nn = sims.argmax(1)
    same = float((sid[nn] == sid).mean())
    counts = np.bincount(sid)
    chance = float(((counts - 1) / (len(sid) - 1) * counts / len(sid)).sum())
    return {"same_session_at_1": same, "chance": chance, "n_turns": int(len(sid))}


def ridge_probe(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, lam: float = 1.0) -> dict:
    """Closed-form ridge classifier on standardised latents; held-out accuracy vs majority."""
    classes = np.unique(y)
    if len(classes) < 2 or len(y) < 10:
        return {"skipped": True}
    idx = rng.permutation(len(y))
    cut = int(len(y) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-5
    xs = (x - mu) / sd
    xb = np.concatenate([xs, np.ones((len(xs), 1))], 1)
    y1 = (y[:, None] == classes[None]).astype(np.float64)
    w = np.linalg.solve(xb[tr].T @ xb[tr] + lam * np.eye(xb.shape[1]), xb[tr].T @ y1[tr])
    pred = classes[(xb[te] @ w).argmax(1)]
    majority = float((y[te] == classes[np.bincount(np.searchsorted(classes, y[tr])).argmax()]).mean())
    return {"accuracy": float((pred == y[te]).mean()), "majority": majority, "n_test": int(len(te))}


def effective_rank(z: np.ndarray) -> float:
    zc = z - z.mean(0)
    s = np.linalg.svd(zc, compute_uv=False)
    p = s**2 / (s**2).sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a ChatJEPA checkpoint")
    ap.add_argument("--checkpoint", default=str(default_checkpoint()))
    ap.add_argument("--data", default=str(turns_jsonl()))
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    loaded = load_checkpoint(args.checkpoint)
    cfg = loaded.cfg
    device = pick_device(args.device)
    level1 = loaded.level1.to(device).eval()
    level2 = loaded.level2.to(device).eval()
    sessions = list(read_jsonl(args.data))
    cc = cfg.conversation
    cache = LatentCache.build(
        level1, sessions, ByteTokenizer(), cfg.model.max_len, cc.head_bytes, cc.tail_bytes, device
    )
    train_cache, hold = cache.split(args.holdout, rng)

    z_all = np.concatenate(cache.latents)
    roles = np.concatenate(cache.roles)
    sources = np.concatenate([np.full(len(s.turns), s.source) for s in sessions])
    report = {
        "checkpoint": args.checkpoint,
        "step": loaded.step,
        "sessions": len(sessions),
        "turns": int(len(z_all)),
        "latent_std": float(z_all.std(0).mean()),
        "effective_rank": effective_rank(z_all),
        "surprise_heldout": surprise_test(level2, hold, cc.max_turns, rng),
        "surprise_train": surprise_test(level2, train_cache, cc.max_turns, rng),
        "retrieval": retrieval_sanity(cache),
        "probe_role": ridge_probe(z_all, roles, rng),
        "probe_source": ridge_probe(z_all, sources, rng),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
