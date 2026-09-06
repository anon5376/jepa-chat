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
