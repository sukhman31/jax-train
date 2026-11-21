import codecs
import pickle
import time

import jax
import jax.numpy as jnp
from flax import nnx

from inference.sampling import sample_token
from tokenizers.bpe_tokenizer import BPETokenizer
from transformer import Model

EOS = '<|endoftext|>'
PROMPT_BUCKETS = (32, 64, 128)
PAD_ID = 0


class InferenceEngine:
    """Loads a checkpoint once and streams generated token ids.

    Every jitted function sees a fixed set of shapes, so nothing recompiles after warmup:
    prompts are right-padded to a bucket and the KV cache is always max_seq_len long.
    """

    def __init__(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.config = checkpoint['config']
        self.tokenizer = BPETokenizer(max_merges=self.config['max_merges'])
        self.tokenizer.pairs_dict = checkpoint['tokenizer_pairs']
        self.tokenizer.vocab = checkpoint['tokenizer_vocab']

        self.rngs = nnx.Rngs(0)
        model = Model(self.config, self.rngs)
        nnx.update(model, checkpoint['model_state'])
        model.eval()
        self.graphdef, self.state = nnx.split(model)

        self.max_len = self.config['max_seq_len']
        self.buckets = tuple(b for b in PROMPT_BUCKETS if b < self.max_len)
        self._prefill = jax.jit(self._prefill_impl)
        self._decode = jax.jit(self._decode_impl)

    def encode(self, prompt):
        # Long prompts keep their end: that's the part the continuation depends on.
        return self.tokenizer.encode(prompt)[-self.buckets[-1]:]

    def warmup(self):
        for bucket in self.buckets:
            for _ in self.stream_ids([PAD_ID] * bucket, max_new_tokens=2):
                pass

    def stream_ids(self, prompt_ids, max_new_tokens=150, temperature=0.8, top_k=50,
                   top_p=0.9, repetition_penalty=1.2, seed=None):
        prompt_len = len(prompt_ids)
        bucket = next(b for b in self.buckets if b >= prompt_len)
        max_new_tokens = min(max_new_tokens, self.max_len - prompt_len)
        padded = jnp.array(prompt_ids + [PAD_ID] * (bucket - prompt_len), dtype=jnp.int32)
        sampling = (jnp.float32(temperature), jnp.int32(top_k),
                    jnp.float32(top_p), jnp.float32(repetition_penalty))

        key = jax.random.PRNGKey(time.time_ns() % (2**31) if seed is None else seed)
        key, sub = jax.random.split(key)
        step = self._prefill(self.state, padded, jnp.int32(prompt_len), sub, *sampling)

        for i in range(max_new_tokens):
            token, cache, history = step
            # Queue the next step before reading this token back: JAX dispatches asynchronously,
            # so the device computes token i+1 while Python decodes and sends token i.
            if i + 1 < max_new_tokens:
                key, sub = jax.random.split(key)
                step = self._decode(self.state, token, cache, jnp.int32(prompt_len + i),
                                    history, sub, *sampling)
            yield int(token)

    def _empty_cache(self):
        num_kv_heads = self.config.get('num_kv_heads', self.config['num_heads'])
        head_dim = self.config['output_dim'] // self.config['num_heads']
        shape = (1, num_kv_heads, self.max_len, head_dim)
        return [(jnp.zeros(shape), jnp.zeros(shape)) for _ in range(self.config['num_of_blocks'])]

    def _prefill_impl(self, state, padded_ids, prompt_len, key,
                      temperature, top_k, top_p, repetition_penalty):
        model = nnx.merge(self.graphdef, state)
        logits, cache, _ = model(padded_ids[None, :], self.rngs, cache=self._empty_cache(), cache_pos=0)
        # Padding sits after the prompt and attention is causal, so logits at prompt_len-1
        # never saw it. The padded cache slots get overwritten before they are ever attended to.
        last_logits = jax.lax.dynamic_index_in_dim(logits[0], prompt_len - 1, keepdims=False)
        history = jnp.zeros(self.max_len, jnp.int32).at[:padded_ids.shape[0]].set(padded_ids)
        token = sample_token(last_logits, history, prompt_len, key,
                             temperature, top_k, top_p, repetition_penalty)
        return token, cache, history.at[prompt_len].set(token)

    def _decode_impl(self, state, token, cache, pos, history, key,
                     temperature, top_k, top_p, repetition_penalty):
        model = nnx.merge(self.graphdef, state)
        logits, cache, _ = model(token.reshape(1, 1), self.rngs, cache=cache, cache_pos=pos)
        next_token = sample_token(logits[0, 0], history, pos + 1, key,
                                  temperature, top_k, top_p, repetition_penalty)
        return next_token, cache, history.at[pos + 1].set(next_token)


class TextStreamer:
    """Turns token ids into text pieces that are safe to show immediately.

    Byte-level BPE tokens can split a UTF-8 character, so bytes go through an incremental
    decoder. Text that could be the start of EOS is held back until it's clearly not.
    """

    def __init__(self, vocab, stop=EOS):
        self.vocab = vocab
        self.stop = stop
        self.stopped = False
        self._utf8 = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self._pending = ''

    def push(self, token_id):
        self._pending += self._utf8.decode(self.vocab.get(token_id, b''))
        stop_at = self._pending.find(self.stop)
        if stop_at != -1:
            self.stopped = True
            text, self._pending = self._pending[:stop_at], ''
            return text
        hold = self._partial_stop_len()
        text = self._pending[:len(self._pending) - hold]
        self._pending = self._pending[len(text):]
        return text

    def finish(self):
        if self.stopped:
            return ''
        text = self._pending + self._utf8.decode(b'', final=True)
        self._pending = ''
        return text

    def _partial_stop_len(self):
        for n in range(min(len(self.stop) - 1, len(self._pending)), 0, -1):
            if self.stop.startswith(self._pending[-n:]):
                return n
        return 0
