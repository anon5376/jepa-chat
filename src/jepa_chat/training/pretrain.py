"""jepa-pretrain: offline pretraining of Level 1 (bytes) and Level 2 (turn latents)."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence

import numpy as np
import torch

from jepa_chat.config import Config
from jepa_chat.core.checkpoint import build_models, load_checkpoint, save_checkpoint
from jepa_chat.core.paths import pretrained_checkpoint, turns_jsonl
from jepa_chat.data.conversation_dataset import LatentCache, turn_text_dataset
from jepa_chat.data.masking import sample_masks
from jepa_chat.data.tokenizer import ByteTokenizer
from jepa_chat.data.turns import Session, read_jsonl
from jepa_chat.models.conversation import ConversationPredictor
from jepa_chat.models.jepa import TextJEPA
from jepa_chat.training.train import ema_momentum, pick_device

Log = Callable[[str], None]


def pretrain_level1(
    cfg: Config,
    sessions: Sequence[Session],
    model: TextJEPA,
    steps: int,
    device: torch.device,
    log: Log = print,
    seed: int = 0,
) -> float:
    """Span-masked latent prediction over the flattened chat text. Returns final latent_std."""
    gen = torch.Generator().manual_seed(seed)
    dataset = turn_text_dataset(sessions, cfg.train.seq_len)
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    t0, latent_std = time.time(), 0.0
    for step in range(1, steps + 1):
        ids, attn = dataset.sample_batch(cfg.train.batch_size, gen)
        masks = sample_masks(attn, cfg.mask, gen)
        out = model(
            ids.to(device),
            attn.to(device),
            masks.context_mask.to(device),
            masks.target_mask.to(device),
        )
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        model.update_target_encoder(ema_momentum(step, steps, cfg.train.ema_start, cfg.train.ema_end))
        latent_std = out.latent_std.item()
        if step % cfg.train.log_every == 0 or step == 1:
            log(f"[L1] step {step:5d} loss {out.loss.item():.4f} latent_std {latent_std:.4f} ({time.time() - t0:.0f}s)")
    return latent_std


@torch.no_grad()
def evaluate_level2(
    model: ConversationPredictor, cache: LatentCache, max_turns: int, rng: np.random.Generator
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    n = min(64, max(8, len(cache.latents)))
    z, roles, attn = cache.sample_batch(n, max_turns, rng)
    loss, stats = model.loss(z.to(device), roles.to(device), attn.to(device))
    model.train()
    return stats


def pretrain_level2(
    cfg: Config,
    sessions: Sequence[Session],
    level1: TextJEPA,
    level2: ConversationPredictor,
    steps: int,
    device: torch.device,
    log: Log = print,
    holdout: float = 0.1,
    seed: int = 0,
    batch_size: int = 32,
    lr: float = 3e-4,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    level1.to(device).eval()
    tok = ByteTokenizer()
    cc = cfg.conversation
    cache = LatentCache.build(level1, sessions, tok, cfg.model.max_len, cc.head_bytes, cc.tail_bytes, device)
    train_cache, hold_cache = cache.split(holdout, rng)
    log(f"[L2] cache: {len(train_cache.latents)} train / {len(hold_cache.latents)} held-out")
    level2.to(device).train()
    opt = torch.optim.AdamW(level2.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    t0, last = time.time(), {}
    for step in range(1, steps + 1):
        z, roles, attn = train_cache.sample_batch(batch_size, cc.max_turns, rng)
        loss, stats = level2.loss(z.to(device), roles.to(device), attn.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(level2.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1 or step == steps:
            held = evaluate_level2(level2, hold_cache, cc.max_turns, rng) if hold_cache.latents else {}
            last = {"train_" + k: v for k, v in stats.items()} | {"held_" + k: v for k, v in held.items()}
            log(
                f"[L2] step {step:5d} loss {stats['loss2']:.4f} ratio {stats['ratio']:.3f}"
                + (f" | held loss {held['loss2']:.4f} ratio {held['ratio']:.3f}" if held else "")
                + f" ({time.time() - t0:.0f}s)"
            )
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description="Pretrain ChatJEPA levels from turns.jsonl")
    ap.add_argument("--data", default=str(turns_jsonl()))
    ap.add_argument("--config", default=None, help="YAML config (defaults: max_len 512)")
    ap.add_argument("--stage", choices=["level1", "level2", "all"], default="all")
    ap.add_argument("--init", default=None, help="checkpoint to start from (required for --stage level2)")
    ap.add_argument("--out", default=str(pretrained_checkpoint()))
    ap.add_argument("--steps1", type=int, default=3000)
    ap.add_argument("--steps2", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sessions = list(read_jsonl(args.data))
    if not sessions:
        raise SystemExit(f"no sessions in {args.data}; run jepa-ingest first")
    device = pick_device(args.device)

    if args.init:
        loaded = load_checkpoint(args.init)
        cfg, level1, level2 = loaded.cfg, loaded.level1, loaded.level2
        pre_std = loaded.pretrain_latent_std
    else:
        cfg = Config.from_yaml(args.config) if args.config else Config()
        if not args.config:
            cfg.model.max_len = 512
            cfg.train.seq_len = 256
            cfg.train.batch_size = 64
        level1, level2 = build_models(cfg)
        pre_std = None
    print(f"device={device} sessions={len(sessions)} turns={sum(len(s.turns) for s in sessions)}")

    if args.stage in ("level1", "all"):
        pre_std = pretrain_level1(cfg, sessions, level1, args.steps1, device, seed=args.seed)
    if args.stage in ("level2", "all"):
        if args.stage == "level2" and not args.init:
            raise SystemExit("--stage level2 needs --init <checkpoint with a trained level1>")
        stats = pretrain_level2(cfg, sessions, level1, level2, args.steps2, device, seed=args.seed)
        print("final:", {k: round(v, 4) for k, v in stats.items()})
    out = save_checkpoint(args.out, cfg, level1, level2, pretrain_latent_std=pre_std)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
