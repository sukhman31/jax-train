import pickle
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from transformer import Model
from tokenizers.bpe_tokenizer import BPETokenizer


class Generator(nnx.Module):
    def __init__(self, checkpoint_path, rngs: nnx.Rngs):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.config = checkpoint['config']
        self.rngs = rngs

        self.tokenizer = BPETokenizer(max_merges=self.config['max_merges'])
        self.tokenizer.pairs_dict = checkpoint['tokenizer_pairs']
        self.tokenizer.vocab = checkpoint['tokenizer_vocab']

        self.model = Model(self.config, rngs)
        nnx.update(self.model, checkpoint['model_state'])
        self.model.eval()

    def _init_cache(self, batch_size, max_len):
        num_blocks = self.config['num_of_blocks']
        num_kv_heads = self.config.get('num_kv_heads', self.config['num_heads'])
        head_dim = self.config['output_dim'] // self.config['num_heads']
        cache = []
        for _ in range(num_blocks):
            k = jnp.zeros((batch_size, num_kv_heads, max_len, head_dim))
            v = jnp.zeros((batch_size, num_kv_heads, max_len, head_dim))
            cache.append((k, v))
        return cache

    def _decode_single(self, graphdef, state, token_id, cache, cache_pos):
        model = nnx.merge(graphdef, state)
        x = token_id.reshape(1, 1)
        logits, new_cache = model(x, self.rngs, cache=cache, cache_pos=cache_pos)
        next_id = jnp.argmax(logits[0, 0, :]).astype(jnp.int32)
        return next_id, new_cache

    def generate(self, prompt, max_new_tokens=100, temperature=0.8):
        token_ids = self.tokenizer.encode(prompt)
        prompt_len = len(token_ids)
        max_len = prompt_len + max_new_tokens

        cache = self._init_cache(1, max_len)

        graphdef, state = nnx.split(self.model)

        @jax.jit
        def prefill(state, token_ids_arr, cache):
            model = nnx.merge(graphdef, state)
            x = token_ids_arr.reshape(1, -1)
            logits, new_cache = model(x, self.rngs, cache=cache, cache_pos=0)
            first_id = jnp.argmax(logits[0, -1, :]).astype(jnp.int32)
            return first_id, new_cache

        @jax.jit
        def decode_loop(state, first_id, cache):
            generated = jnp.zeros(max_new_tokens, dtype=jnp.int32)
            generated = generated.at[0].set(first_id)

            def body(i, carry):
                gen_tokens, c, prev_id = carry
                next_id, new_c = self._decode_single(graphdef, state, prev_id, c, prompt_len + i)
                gen_tokens = gen_tokens.at[i].set(next_id)
                return gen_tokens, new_c, next_id

            generated, final_cache, _ = jax.lax.fori_loop(
                1, max_new_tokens, body, (generated, cache, first_id)
            )
            return generated

        token_ids_arr = jnp.array(token_ids, dtype=jnp.int32)
        first_id, cache = prefill(state, token_ids_arr, cache)
        generated = decode_loop(state, first_id, cache)

        token_ids.extend(generated.tolist())
        return self.tokenizer.decode(token_ids)
