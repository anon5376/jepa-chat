# Architecture notes

## Why JEPA for a chat assistant

Autoregressive LLMs model text by predicting the next token. Every unpredictable
surface detail costs capacity, and there is no explicit internal state that
represents "what the conversation is about" apart from the token history.

A JEPA instead learns an **encoder** whose outputs are predictable from partial
context, and a **predictor** that works purely in that latent space. Loss is
measured between predicted and target embeddings, never on tokens. The target
encoder is an exponential moving average (EMA) of the online encoder, which
(together with the asymmetric predictor) prevents the trivial collapsed solution.

The bet: a latent model of the conversation is a better *core* for an assistant,
and language models are better used as **decoders and specialist experts** on
top of it than as the whole system.

## Components

```
user text ──► ByteTokenizer ──► online encoder ──► context latents ──┐
                                                                     ├─► predictor ─► predicted target latents
                              target encoder (EMA) ─► target latents ┘         │
                                                                               ▼  loss in latent space
```

Inference path (current placeholder):

```
prompt ──► JEPA.embed ──► latent summary ──► MixtureRouter ──► LLMBackend ──► reply
```

- `TextEncoder`: pre-norm transformer encoder. Same class for online and target.
- `LatentPredictor`: narrower transformer. Context latents are projected down,
  target positions replaced with a learned mask token plus positional embedding,
  output projected back up to encoder width.
- `TextJEPA`: wires both, exposes `forward` (training loss) and `embed` (pooled latent).
- Masking: I-JEPA style contiguous target spans on a 1-D token sequence.
- `MixtureRouter`: picks an `LLMBackend` per turn. Round-robin now; learned later.

## Planned phases

1. **Stability** — toy corpus, byte tokens. Confirm loss falls and `latent_std` stays
   healthy. Try VICReg-style variance/covariance terms only if collapse appears.
2. **Scale the encoder** — BPE tokenizer, real corpus, longer context.
3. **Decoder** — train a small autoregressive decoder conditioned on JEPA latents
   (prefix tokens or cross-attention). Compare against nearest-token baseline.
4. **Conversation world model** — treat turns as the "patches": predict the latent
   of the assistant's next turn from the latent history. This is the actual chat core.
5. **Mixture of experts / LLMs** — router over latents chooses among local models,
   hosted APIs, retrieval, and tools. Measure whether the latent router beats
   prompt-based routing.
6. **Hierarchical JEPA** — stack a slower JEPA over conversation-level latents
   for long-horizon planning (H-JEPA idea).

## Open questions

- Is byte-level masking too easy (local n-gram prediction) to force semantic latents?
  Longer spans and multi-span targets are the first knob.
- What does "the latent of the assistant's next turn" mean when the assistant is
  ourselves? Options: latent of a reference model's reply, or of the user's *reaction*.
- How to evaluate latent quality without a decoder: linear probes on topic, sentiment,
  intent; retrieval of paraphrases.

---

# ChatJEPA: "one mind, many mouths"

The chat prototype. A local **JEPA core** owns conversation state and long-term latent
memory. The LLM apps you already pay for (Claude Code, Claude Desktop, Cursor) are
interchangeable, stateless **mouths** that talk to the core over MCP. No API keys.

```
Claude Code ──HTTP MCP──┐                                        ┌── jepa-tui  (Textual, SSE)
Cursor ──────HTTP MCP───┼──► jepa-core  127.0.0.1:8765  /mcp /api ┤
Claude Desktop ─stdio───┘   (jepa-mcp proxy)                     └── hooks/observe.py (Claude Code)
                                       │
            ModelThread (MPS): Level 1 TextJEPA + Level 2 ConversationPredictor, online training
            CPU InferenceSnapshot, swapped every N steps, serves every request
            MemoryStore: ~/.jepa-chat/memory.sqlite + whitened cosine index
```

## Two levels

- **Level 1, turn encoder**: the byte-level `TextJEPA`. A turn (head 768 + tail 256 bytes)
  becomes one latent via the EMA target encoder. Trained with span masking plus a
  VICReg-style variance/covariance term so latents cannot shrink into a cone.
- **Level 2, conversation predictor**: a causal transformer over turn latents and role
  embeddings. From turns ≤ k it predicts the normalised latent of turn k+1. Level 1 latents
  are detached, so this level cannot collapse by moving its own targets. Its hidden state is
  the *conversation state*; the prediction error on the next real turn is the **surprise**
  (reported as a ratio against error vs random memory latents: 0 = predicted, ~1 = chance).

## Online learning

One thread owns the accelerator. Each step samples replay windows from memory (25% biased
to sessions touched in the last hour), takes a Level 1 span-mask step on the turns' bytes
and a Level 2 step on their stored latents, and updates the EMA target encoder with a
constant momentum. Inference never waits on training: it runs on a CPU snapshot that is
swapped every `snapshot_every` steps. Stored latents carry `enc_version`; stale ones are
re-embedded periodically. A `latent_std` watchdog pauses training on collapse.

## MCP surface

| tool | purpose |
|---|---|
| `jepa_observe(role, text, mouth)` | record a turn; returns surprise + related memories |
| `jepa_state()` | turns, mouths seen, surprise trend, top memories, training stats |
| `jepa_recall(query, k)` | related past turns across sessions and apps |
| `jepa_predict_next()` | nearest past user turns to the predicted next latent |
| `jepa_plan(candidates)` | latent planning: rank draft replies by fit to the predicted next latent, preview imagined user reaction |
| `jepa_session(new/resume/list/current)` | switch conversations |
| `jepa_stats()`, `jepa_control(pause/resume/checkpoint/reembed)` | training |

Prompt `jepa_briefing` and resource `jepa://state`. In Claude Code, `UserPromptSubmit` and
`Stop` hooks post every turn to `/api/observe`, so observation does not depend on the model
remembering to call a tool.

## Data

`jepa-ingest` unifies: official ChatGPT export, Claude.ai export, local Claude Code
transcripts, Codex CLI rollouts, Cursor IDE chats (`state.vscdb`), Cursor CLI transcripts,
and markdown transcripts. Everything stays under `~/.jepa-chat`.
