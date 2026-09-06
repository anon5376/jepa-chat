# 001 — Toy text JEPA

**Question:** does a byte-level text JEPA train stably (loss falls, latents do not collapse)
on a tiny corpus with EMA target encoder and span masking?

**Setup:** see `config.yaml`. Built-in sample paragraph unless `train.corpus_path` is set.

**Run:**

```bash
uv run jepa-train --config experiments/001_text_jepa_toy/config.yaml
```

**What to watch:** `loss` should decrease; `latent_std` should stay clearly above zero.
A `latent_std` that drifts toward 0 means representational collapse.

**Result:** _not run yet_
