"""Minimal training loop for TextJEPA."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from jepa_chat.config import Config
from jepa_chat.data.datasets import ChunkedTextDataset, load_corpus
from jepa_chat.data.masking import sample_masks
from jepa_chat.data.tokenizer import ByteTokenizer
from jepa_chat.models.jepa import TextJEPA


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ema_momentum(step: int, total: int, start: float, end: float) -> float:
    # cosine schedule from start -> end, as in I-JEPA
    progress = min(step / max(total, 1), 1.0)
    return end - (end - start) * (1 + math.cos(math.pi * progress)) / 2


def train(cfg: Config) -> Path:
    torch.manual_seed(cfg.train.seed)
    gen = torch.Generator().manual_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)

    tokenizer = ByteTokenizer()
    dataset = ChunkedTextDataset(load_corpus(cfg.train.corpus_path), cfg.train.seq_len, tokenizer)
    model = TextJEPA(cfg.model).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    print(f"device={device} params={sum(p.numel() for p in params):,} steps={cfg.train.steps}")
    t0 = time.time()
    for step in range(1, cfg.train.steps + 1):
        ids, attn = dataset.sample_batch(cfg.train.batch_size, gen)
        masks = sample_masks(attn, cfg.mask, gen)
        ids, attn = ids.to(device), attn.to(device)
        ctx_mask, tgt_mask = masks.context_mask.to(device), masks.target_mask.to(device)

        out = model(ids, attn, ctx_mask, tgt_mask)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        model.update_target_encoder(
            ema_momentum(step, cfg.train.steps, cfg.train.ema_start, cfg.train.ema_end)
        )

        if step % cfg.train.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(
                f"step {step:5d} loss {out.loss.item():.4f} "
                f"latent_std {out.latent_std.item():.4f} ({elapsed:.1f}s)"
            )

    ckpt = Path(cfg.train.checkpoint_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.to_dict(), "model": model.state_dict()}, ckpt)
    print(f"saved {ckpt}")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a toy TextJEPA.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--steps", type=int, default=None, help="override train.steps")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.steps is not None:
        cfg.train.steps = args.steps
    train(cfg)


if __name__ == "__main__":
    main()
