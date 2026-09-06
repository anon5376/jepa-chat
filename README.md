# jepa-chat

Experimental research toward a ChatGPT-style assistant whose core is a
**Joint-Embedding Predictive Architecture (JEPA)** rather than an autoregressive
token predictor, with a **mixture of LLMs** and other components used as decoders,
tools, and specialist experts around that latent core.

This is a research sandbox, not a product. Expect rough edges, toy scales, and
frequent rewrites. Design notes live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md);
running notes in [docs/research_log.md](docs/research_log.md).

## Layout

```
src/jepa_chat/
  config.py            dataclass configs + YAML loading
  data/                byte tokenizer, span masking, tiny corpus loader
  models/              TextEncoder, LatentPredictor, TextJEPA (online + EMA target)
  training/train.py    minimal training loop (MPS / CUDA / CPU)
  inference/
    decoder.py         latent -> text (placeholder decoder)
    llm_mixture.py     LLM backend protocol + router
    chat.py            CLI chat loop
experiments/           one folder per experiment: config + README with results
notebooks/             scratch notebooks
tests/                 fast unit tests
docs/                  architecture notes and research log
```

## Quick start

```bash
uv sync --extra dev
uv run pytest -q
uv run jepa-train --config experiments/001_text_jepa_toy/config.yaml
uv run jepa-chat
```

`jepa-chat` runs with an offline echo backend by default, so it works without any
API keys. Provider adapters plug in through the `LLMBackend` protocol in
`src/jepa_chat/inference/llm_mixture.py`.

## Status

- [x] Text JEPA skeleton: byte tokenizer, span masking, online/EMA encoders, predictor
- [x] Training loop with collapse monitor (`latent_std`)
- [x] Mixture router + CLI chat loop (echo backend)
- [ ] Experiment 001: verify stable training on a toy corpus
- [ ] Real tokenizer and a real corpus
- [ ] Latent-conditioned decoder (LLM reads JEPA latents)
- [ ] Learned router over JEPA latents across several LLM backends
- [ ] Multi-turn conversation world model

## License

MIT
