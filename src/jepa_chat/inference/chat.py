"""Command-line chat loop: JEPA latent summary -> router -> LLM backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from jepa_chat.config import Config, ModelConfig
from jepa_chat.data.tokenizer import ByteTokenizer
from jepa_chat.inference.llm_mixture import EchoBackend, MixtureRouter
from jepa_chat.models.jepa import TextJEPA


def load_model(path: str | Path | None) -> TextJEPA | None:
    if path is None or not Path(path).exists():
        return None
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"]["model"])
    model = TextJEPA(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def run_chat(model: TextJEPA | None, router: MixtureRouter, max_len: int) -> None:
    tokenizer = ByteTokenizer()
    history: list[tuple[str, str]] = []
    print("jepa-chat (type 'exit' to quit)")
    if model is None:
        print("no checkpoint found; running without a JEPA latent")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        latent = None
        if model is not None:
            ids, attn = tokenizer.batch_encode([prompt], max_len)
            latent = model.embed(ids, attn)[0]
        name, reply = router.generate(latent, prompt, history)
        history.append((prompt, reply))
        suffix = f" | latent norm {latent.norm():.2f}" if latent is not None else ""
        print(f"{name}> {reply}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the JEPA + LLM mixture prototype.")
    parser.add_argument("--checkpoint", default=Config().train.checkpoint_path)
    args = parser.parse_args()
    model = load_model(args.checkpoint)
    max_len = model.cfg.max_len if model is not None else ModelConfig().max_len
    router = MixtureRouter(backends=[EchoBackend()])
    run_chat(model, router, max_len)


if __name__ == "__main__":
    main()
