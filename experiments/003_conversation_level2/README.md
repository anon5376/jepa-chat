# 003 — Level 2 conversation predictor

**Question:** can a small causal transformer over turn latents predict the latent of the
next turn better than chance (shuffled-target baseline) on held-out sessions?

**Run:** `uv run jepa-pretrain --stage all` then `uv run jepa-eval`.

**Metric:** `ratio` = loss vs true next turn / loss vs shuffled targets. < 1 means real signal;
target < 0.8 on held-out sessions.

**Result:** _not run yet_
