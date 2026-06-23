# JAX LLM — From Scratch to Inference

**Goal:** Build a modern LLM from scratch in JAX, understand every technique used in production inference systems like the GLM-5.2 deployment, and run real inference on a model you trained yourself.

---

## Where You Are Now

You have already built and understood:

- [x] JAX fundamentals — `grad`, `vmap`, `jit`, `lax`, pytrees
- [x] Flax NNX — modules, params, `nnx.jit`, `nnx.value_and_grad`
- [x] Embedding layer with learned positional encodings
- [x] Scaled dot product attention with causal mask
- [x] Multi-head attention

---

## The Roadmap

### Phase 1 — Complete the GPT Architecture
*Finish porting batran-lib to JAX. Everything here is architecture-level — no new concepts, just assembling what you know.*

- [x] `LayerNorm` — normalize activations per token
- [x] `FeedForward` — two linear layers with GELU activation
- [x] `GPTBlock` — MHA + LayerNorm + FeedForward + residual connections
- [x] `GPTModel` — stack of blocks + embedding + output head
- [ ] End-to-end forward pass: tokens → logits

**Resources:**
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — the original paper with line-by-line code
- [Andrej Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT) — minimal GPT implementation to compare against
- [Karpathy's Zero to Hero lecture on GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — best video walkthrough of GPT internals

---

### Phase 2 — Training Loop on Real Data
*Get the model actually learning something on real text.*

- [ ] BPE tokenizer (use `tiktoken` or implement basic version)
- [ ] Data pipeline — load text, tokenize, create sliding window batches
- [ ] Loss function — cross-entropy on next-token prediction
- [ ] Training loop with `nnx.jit` + `optax`
- [ ] Learning rate schedule — warmup + cosine decay
- [ ] Gradient clipping — prevent exploding gradients
- [ ] Checkpoint saving/loading

**Resources:**
- [Karpathy — minBPE](https://github.com/karpathy/minbpe) — minimal BPE tokenizer implementation
- [The Pile dataset](https://pile.eleuther.ai/) — open training data
- [Optax documentation](https://optax.readthedocs.io/) — JAX optimizers including schedules

---

### Phase 3 — Modern Architecture Upgrades
*Replace the basic GPT components with what modern models like LLaMA/GLM actually use.*

- [ ] **RoPE** (Rotary Positional Embeddings) — replaces learned positional encodings, generalises to longer sequences
- [ ] **RMSNorm** — simpler and faster than LayerNorm, used in LLaMA/GLM
- [ ] **SwiGLU** activation — replaces GELU in feedforward, used in LLaMA
- [ ] **GQA** (Grouped Query Attention) — fewer KV heads than Q heads, reduces memory, used in LLaMA 3
- [ ] **MoE** (Mixture of Experts) — only activate a fraction of parameters per token, used in GLM-5.2

**Resources:**
- [RoFormer paper — RoPE](https://arxiv.org/abs/2104.09864)
- [LLaMA 2 paper](https://arxiv.org/abs/2307.09288) — describes RMSNorm, SwiGLU, GQA all in one place
- [LLaMA from scratch — Umar Jamil](https://www.youtube.com/watch?v=oM4VmoabDAI) — best video on modern architecture components

---

### Phase 4 — Inference Optimisations
*This is where you close the gap with the GLM-5.2 article. These are the techniques that make inference fast.*

- [ ] **KV Cache** — store past keys/values so you don't recompute them on every token. Biggest single speedup for autoregressive generation.
- [ ] **Sampling strategies** — greedy, temperature, top-p, top-k
- [ ] **Batched inference** — generate for multiple prompts simultaneously
- [ ] **Speculative decoding / MTP** — generate draft tokens with a small head, verify with the main model. Used in GLM-5.2.
- [ ] **INT8 quantization** — compress weights from float32 to int8, halving memory and speeding up compute

**Resources:**
- [Efficient Inference paper — KV Cache explained](https://arxiv.org/abs/2307.08691)
- [Speculative decoding paper](https://arxiv.org/abs/2211.17192)
- [Tim Dettmers — quantization explainer](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)

---

### Phase 5 — Scale and Systems (Optional / Advanced)
*This is what Baseten built. Requires multiple GPUs but the concepts are fully learnable.*

- [ ] **`jax.pmap`** — distribute computation across multiple GPUs/TPUs
- [ ] **Prefill/decode disaggregation** — run input processing and token generation on separate workers
- [ ] **NVFP4/INT4 quantization** — further compress weights for Blackwell GPUs

**Resources:**
- [JAX distributed training guide](https://jax.readthedocs.io/en/latest/multi_process.html)
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) — the toolkit Baseten used for disaggregation
- [vLLM paper — PagedAttention](https://arxiv.org/abs/2309.06180) — industry standard for production inference

---

## What Each Phase Teaches You

| Phase | What you will understand from the GLM-5.2 article |
|---|---|
| 1 — GPT Architecture | The base transformer GLM-5.2 is built on |
| 2 — Training | How any LLM acquires its knowledge |
| 3 — Modern Architecture | MoE (40B active of 744B params), RoPE for 1M context |
| 4 — Inference | KV cache, MTP speculative decoding, quantization |
| 5 — Systems | Disaggregation, KV-aware routing, production scale |

---

## The Honest Ceiling

With a single GPU you can fully implement Phases 1–4 and understand every concept in Phase 5. The only thing hardware stops you from doing is running a 744B parameter model. The knowledge gap between you and Baseten is zero — the hardware gap is about $10M of GPUs.

---

## Key Papers (In Reading Order)

1. [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — the original transformer
2. [GPT-3 (2020)](https://arxiv.org/abs/2005.14165) — GPT at scale
3. [LLaMA 2 (2023)](https://arxiv.org/abs/2307.09288) — modern architecture: RMSNorm, SwiGLU, GQA
4. [FlashAttention (2022)](https://arxiv.org/abs/2205.14135) — memory-efficient attention
5. [Speculative Decoding (2022)](https://arxiv.org/abs/2211.17192) — MTP / speculation
6. [vLLM / PagedAttention (2023)](https://arxiv.org/abs/2309.06180) — production inference

---

## Communities

- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) — practitioners running and optimising LLMs
- [Hugging Face Discord](https://huggingface.co/join/discord) — open source ML community
- [JAX GitHub Discussions](https://github.com/google/jax/discussions) — JAX-specific questions
- [EleutherAI](https://www.eleuther.ai/) — open source LLM research community
