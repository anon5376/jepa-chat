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
  data/                byte tokenizer, span masking, turns/sessions, latent datasets
  models/              TextJEPA (level 1), ConversationPredictor (level 2)
  ingest/              parsers: ChatGPT, Claude.ai, Claude Code, Codex, Cursor, markdown
  training/            train (toy), pretrain (both levels), evaluate
  core/                daemon: memory store, model thread, service, HTTP/SSE app, CLI
  mcp/                 MCP server, stdio proxy, client install snippets + hook
  tui/                 Textual dashboard
  inference/           echo backend router + toy CLI chat
experiments/           one folder per experiment: config + README with results
notebooks/             scratch notebooks
tests/                 fast unit tests
docs/                  architecture notes and research log
```

## ChatJEPA quick start ("one mind, many mouths")

A local JEPA core holds conversation state and long-term latent memory; Claude Code,
Claude Desktop and Cursor are interchangeable mouths talking to it over MCP. No API keys.

```bash
uv sync --extra dev
# 1. build a corpus from what is already on the laptop (+ optional official exports)
uv run jepa-ingest --all-local --chatgpt ~/Downloads/chatgpt-export.zip --claude ~/Downloads/claude-export.zip
# 2. pretrain both levels, then check it learned something
uv run jepa-pretrain --config experiments/002_chat_level1/config.yaml --stage all
uv run jepa-eval
# 3. run the core, load memory, open the dashboard
uv run jepa-core serve            # terminal 1
uv run jepa-core import-memory    # once
uv run jepa-tui                   # terminal 2
# 4. plug in the mouths (prints config; --write creates hook/rule files)
uv run jepa-core install-client all --write
```

Then in Claude Code: `claude mcp add --transport http --scope user jepa http://127.0.0.1:8765/mcp`.
Design notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#chatjepa-one-mind-many-mouths).

## Toy experiment

```bash
uv run jepa-train --config experiments/001_text_jepa_toy/config.yaml
uv run jepa-chat   # offline echo backend, no keys needed
```

## Status

- [x] Text JEPA skeleton: byte tokenizer, span masking, online/EMA encoders, predictor
- [x] Training loop with collapse monitor (`latent_std`)
- [x] Mixture router + CLI chat loop (echo backend)
- [x] Ingestion of local chat history (Claude Code, Codex, Cursor) and exports
- [x] Two-level model: turn encoder + conversation predictor, offline pretraining
- [x] Core daemon with online learning, latent memory, surprise, MCP tools, TUI
- [ ] Validate surprise on held-out sessions with a larger corpus (see experiment 003)
- [ ] Latent planning: imagine turns ahead, score candidate replies (`jepa_plan`)
- [ ] BPE tokenizer for Level 1; latent-conditioned decoder

## License

MIT
