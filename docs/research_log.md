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
