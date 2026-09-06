# Research log

## 2026-09-06

- Created the repository and scaffold. Byte-level text JEPA skeleton, training loop,
  mixture router with an echo backend, CLI chat loop, tests.
- Next: run experiment 001 and record loss / `latent_std` curves.

## 2026-09-06 (later)

- Built ChatJEPA: two-level model, core daemon, MCP server, TUI, ingestion of local history.
- Corpus v1 (Claude Code + one markdown transcript): 19 sessions / 268 turns. Level 1 byte JEPA
  loss 0.54 -> 0.012 but `latent_std` fell 0.91 -> 0.20 (partial collapse, tiny corpus).
  Level 2 train ratio 0.78, held-out ratio ~1.0 with a single held-out session: no evidence of
  generalisation at this scale. Recall cosines were all ~0.999 (latents in a cone).
- Fixes: whitened cosine index in memory; VICReg variance/covariance term on Level 1.
- Corpus v2 (+ Codex rollouts, Cursor chats): 228 sessions / 8,379 turns, median 3 KB/turn.
  Raised Level 1 context to 1024 bytes (head 768 + tail 256). Pretraining rerun pending.
- Corpus v2 pretraining: Level 1 stable (`latent_std` ~1.04). Level 2 held-out surprise ratio 0.50
  (vs same-session) / 0.40 (vs other), AUROC 0.75 / 0.84 on 46 held-out sessions. First real
  evidence the conversation predictor generalises. Details in experiments/003.
- Byte latent is a session/style embedding, not a topic embedding: recall went hybrid (latent + BM25).
- Latent planning tool `jepa_plan` added. Full demo path verified: daemon, 8.4k-turn memory,
  proxy, TUI with two mouths on one session.

## 2026-09-07

- Overnight the daemon ran 82k online steps on the fixed 8.4k-turn corpus. Result vs the fresh
  pretrained checkpoint: effective rank 18 -> 12, same-session@1 0.59 -> 0.54, surprise ratio
  0.50 -> 0.68 (Level 2 got worse even with contaminated held-out). Unbounded replay of a static
  corpus degrades the latents; online learning must be tied to new experience.
- Fix: a step budget. Each observed live turn earns `steps_per_new_turn` (20) steps; idle
  replay trickles at `idle_steps_per_min` (2); bank capped at 400. Also: `jepa-pretrain` now
  writes `pretrained.pt` (immutable) and the daemon only ever writes `core.pt`; `jepa-core reset`
  restores the pretrained weights.
