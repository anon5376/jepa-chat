# 003 — Level 2 conversation predictor

**Question:** can a small causal transformer over turn latents predict the latent of the
next turn better than chance (shuffled-target baseline) on held-out sessions?

**Run:** `uv run jepa-pretrain --stage all` then `uv run jepa-eval`.

**Metric:** `ratio` = loss vs true next turn / loss vs shuffled targets. < 1 means real signal;
target < 0.8 on held-out sessions.

**Result (2026-09-06, corpus v2: 228 sessions / 8,379 turns, 20% held-out = 46 sessions):**

| metric | held-out | train |
|---|---|---|
| error ratio, true next turn vs random same-session turn | **0.50** | 0.28 |
| error ratio, true next turn vs random other-session turn | **0.40** | 0.23 |
| AUROC true vs same-session | 0.75 | 0.84 |
| AUROC true vs other-session | 0.84 | 0.94 |

The conversation predictor generalises to unseen sessions: its error on the true next turn is
half its error on a shuffled turn. Pretraining log: train ratio 0.24, held-out 0.77 at 3k steps
(held-out ratio was 0.67 at 2k, so a little overfit; early stopping is a cheap win).

Latent quality (Level 1, whitened): nearest neighbour in the same session 59% vs 18% chance;
role probe 94% vs 50% majority; source probe no better than majority (89%). Effective rank 18/128.

Practical finding: the byte-level latent captures session identity and style, not topic. Recall
by subject was poor until blended with lexical BM25 (`lexical_weight=0.5`), which is now the
default in `jepa_recall`. Live short chatty turns get high surprise (out of distribution vs long
Codex turns) until online learning adapts.
