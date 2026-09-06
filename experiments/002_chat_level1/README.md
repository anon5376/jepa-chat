# 002 — Level 1 byte JEPA on real chat text

**Question:** does the byte JEPA train stably on the user's own chat corpus with longer
target spans (15–35%) and 512-byte context?

**Run:** `uv run jepa-pretrain --config experiments/002_chat_level1/config.yaml --stage level1 --steps1 3000`

**Result (2026-09-06, corpus v2, max_len 1024, VICReg var 1.0 / cov 0.05, 4000 steps, ~19 min on M5 Pro):**
loss 0.70 -> 0.22, `latent_std` 0.90 -> 1.04 and flat. No collapse. Without the regulariser on the
tiny corpus v1, `latent_std` had fallen 0.91 -> 0.20.
