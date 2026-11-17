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

    def _sample_token(self, logits, generated, num_generated, key,
                       temperature, top_k, top_p, repetition_penalty):
        logits = logits.astype(jnp.float32)

        valid = (jnp.arange(generated.shape[0]) < num_generated).astype(jnp.float32)
        penalty_mask = jnp.zeros_like(logits).at[generated].add(valid)
        penalty_mask = (penalty_mask > 0).astype(jnp.float32)
        logits = jnp.where(
            penalty_mask > 0,
            jnp.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty),
            logits,
        )

        logits = logits / temperature

        sorted_indices = jnp.argsort(-logits)
        sorted_logits = logits[sorted_indices]

        topk_mask = jnp.arange(logits.shape[0]) < top_k
        sorted_logits = jnp.where(topk_mask, sorted_logits, -jnp.inf)

        sorted_probs = jax.nn.softmax(sorted_logits)
        cumulative_probs = jnp.cumsum(sorted_probs)
        nucleus_mask = cumulative_probs - sorted_probs < top_p
        sorted_logits = jnp.where(nucleus_mask, sorted_logits, -jnp.inf)

        probs = jax.nn.softmax(sorted_logits)
        sampled_index = jax.random.categorical(key, jnp.log(probs + 1e-10))
        return sorted_indices[sampled_index].astype(jnp.int32)

    def _decode_single(self, graphdef, state, token_id, cache, cache_pos,
                        generated, num_generated, key,
                        temperature, top_k, top_p, repetition_penalty):
        model = nnx.merge(graphdef, state)
        x = token_id.reshape(1, 1)
        logits, new_cache, _ = model(x, self.rngs, cache=cache, cache_pos=cache_pos)
        next_id = self._sample_token(
            logits[0, 0, :], generated, num_generated, key,
            temperature, top_k, top_p, repetition_penalty,
        )
        return next_id, new_cache

    def generate(self, prompt, max_new_tokens=100, temperature=0.8,
                 top_k=50, top_p=0.9, repetition_penalty=1.2):
        token_ids = self.tokenizer.encode(prompt)
        prompt_len = len(token_ids)
        max_len = prompt_len + max_new_tokens
        total_len = prompt_len + max_new_tokens

        cache = self._init_cache(1, max_len)

        graphdef, state = nnx.split(self.model)

        all_tokens = jnp.zeros(total_len, dtype=jnp.int32)
        all_tokens = all_tokens.at[:prompt_len].set(jnp.array(token_ids, dtype=jnp.int32))

        rng_key = jax.random.PRNGKey(42)

        @jax.jit
        def prefill(state, token_ids_arr, cache, all_tokens, rng_key):
            model = nnx.merge(graphdef, state)
            x = token_ids_arr.reshape(1, -1)
            logits, new_cache, _ = model(x, self.rngs, cache=cache, cache_pos=0)
            key, rng_key = jax.random.split(rng_key)
            first_id = self._sample_token(
                logits[0, -1, :], all_tokens, prompt_len, key,
                temperature, top_k, top_p, repetition_penalty,
            )
            all_tokens = all_tokens.at[prompt_len].set(first_id)
            return first_id, new_cache, all_tokens, rng_key

        @jax.jit
        def decode_loop(state, first_id, cache, all_tokens, rng_key):
            generated = jnp.zeros(max_new_tokens, dtype=jnp.int32)
            generated = generated.at[0].set(first_id)

            def body(i, carry):
                gen_tokens, c, prev_id, all_tok, rk = carry
                key, rk = jax.random.split(rk)
                next_id, new_c = self._decode_single(
                    graphdef, state, prev_id, c, prompt_len + i,
                    all_tok, prompt_len + i, key,
                    temperature, top_k, top_p, repetition_penalty,
                )
                gen_tokens = gen_tokens.at[i].set(next_id)
                all_tok = all_tok.at[prompt_len + i].set(next_id)
                return gen_tokens, new_c, next_id, all_tok, rk

            generated, final_cache, _, _, _ = jax.lax.fori_loop(
                1, max_new_tokens, body, (generated, cache, first_id, all_tokens, rng_key)
            )
            return generated

        token_ids_arr = jnp.array(token_ids, dtype=jnp.int32)
        first_id, cache, all_tokens, rng_key = prefill(state, token_ids_arr, cache, all_tokens, rng_key)
        generated = decode_loop(state, first_id, cache, all_tokens, rng_key)

        token_ids.extend(generated.tolist())
        return self.tokenizer.decode(token_ids)
