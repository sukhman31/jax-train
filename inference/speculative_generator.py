import pickle

import jax
import jax.numpy as jnp
from flax import nnx

from transformer import Model
from tokenizers.bpe_tokenizer import BPETokenizer


class SpeculativeGenerator(nnx.Module):
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

    def _sample_token(self, logits, all_tokens, num_generated, key,
                       temperature, top_k, top_p, repetition_penalty):
        logits = logits.astype(jnp.float32)

        valid = (jnp.arange(all_tokens.shape[0]) < num_generated).astype(jnp.float32)
        penalty_mask = jnp.zeros_like(logits).at[all_tokens].add(valid)
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

    def generate(self, prompt, max_new_tokens=100, temperature=0.8,
                 top_k=50, top_p=0.9, repetition_penalty=1.2,
                 draft_layers=2, num_speculative=4):
        token_ids = self.tokenizer.encode(prompt)
        prompt_len = len(token_ids)
        max_len = prompt_len + max_new_tokens
        total_len = prompt_len + max_new_tokens
        num_blocks = self.config['num_of_blocks']

        cache = self._init_cache(1, max_len)

        graphdef, state = nnx.split(self.model)

        all_tokens = jnp.zeros(total_len, dtype=jnp.int32)
        all_tokens = all_tokens.at[:prompt_len].set(jnp.array(token_ids, dtype=jnp.int32))

        rng_key = jax.random.PRNGKey(42)

        @jax.jit
        def prefill(state, token_ids_arr, cache, rng_key, all_tokens):
            model = nnx.merge(graphdef, state)
            x = token_ids_arr.reshape(1, -1)
            logits, new_cache, _ = model(x, self.rngs, cache=cache, cache_pos=0)
            key, rng_key = jax.random.split(rng_key)
            first_id = self._sample_token(
                logits[0, -1, :], all_tokens, prompt_len, key,
                temperature, top_k, top_p, repetition_penalty,
            )
            return first_id, new_cache, rng_key

        @jax.jit
        def draft_one(state, token_id, draft_cache, cache_pos, all_tokens, num_gen, rng_key):
            model = nnx.merge(graphdef, state)
            x = token_id.reshape(1, 1)
            logits, new_draft_cache, _ = model(
                x, self.rngs, cache=draft_cache, cache_pos=cache_pos,
                exit_after=draft_layers,
            )
            probs = jax.nn.softmax(logits[0, 0, :].astype(jnp.float32))
            key, rng_key = jax.random.split(rng_key)
            token = self._sample_token(
                logits[0, 0, :], all_tokens, num_gen, key,
                temperature, top_k, top_p, repetition_penalty,
            )
            token_prob = probs[token]
            return token, token_prob, new_draft_cache, rng_key

        def verify_batch(state, draft_tokens, full_cache, cache_pos):
            model = nnx.merge(graphdef, state)
            x = draft_tokens.reshape(1, -1)
            logits, new_cache, _ = model(
                x, self.rngs, cache=full_cache, cache_pos=cache_pos,
            )
            logits_f = logits[0].astype(jnp.float32)
            probs = jax.nn.softmax(logits_f)
            return probs, new_cache, logits_f

        token_ids_arr = jnp.array(token_ids, dtype=jnp.int32)
        first_id, cache, rng_key = prefill(state, token_ids_arr, cache, rng_key, all_tokens)

        all_tokens = all_tokens.at[prompt_len].set(first_id)
        pos = prompt_len + 1
        generated = [int(first_id)]

        while len(generated) < max_new_tokens:
            draft_cache = []
            for i in range(num_blocks):
                if i < draft_layers:
                    draft_cache.append(cache[i])
                else:
                    k_shape = cache[i][0].shape
                    draft_cache.append((jnp.zeros(k_shape), jnp.zeros(k_shape)))

            draft_tokens = []
            draft_probs = []
            prev_token = all_tokens[pos - 1]
            dc = draft_cache

            for s in range(num_speculative):
                if pos + s >= total_len:
                    break
                token, prob, dc, rng_key = draft_one(
                    state, prev_token, dc, pos + s,
                    all_tokens, pos + s, rng_key,
                )
                draft_tokens.append(int(token))
                draft_probs.append(float(prob))
                prev_token = token
                all_tokens = all_tokens.at[pos + s].set(token)

            if not draft_tokens:
                break

            num_draft = len(draft_tokens)
            draft_arr = jnp.array(draft_tokens, dtype=jnp.int32)

            main_probs, verified_cache, main_logits = verify_batch(
                state, draft_arr, cache, pos,
            )

            accepted = 0
            for i in range(num_draft):
                main_p = float(main_probs[i, draft_tokens[i]])
                draft_p = draft_probs[i]
                ratio = main_p / max(draft_p, 1e-10)

                rng_key, accept_key = jax.random.split(rng_key)
                r = float(jax.random.uniform(accept_key))

                if r < min(1.0, ratio):
                    accepted += 1
                else:
                    rng_key, resample_key = jax.random.split(rng_key)
                    resampled = self._sample_token(
                        main_logits[i], all_tokens, pos + i, resample_key,
                        temperature, top_k, top_p, repetition_penalty,
                    )
                    all_tokens = all_tokens.at[pos + i].set(resampled)
                    draft_tokens[i] = int(resampled)
                    accepted += 1
                    break

            cache = []
            for layer_i in range(num_blocks):
                if layer_i < len(verified_cache):
                    ck, cv = verified_cache[layer_i]
                    cache.append((ck, cv))
                else:
                    cache.append((jnp.zeros_like(ck), jnp.zeros_like(cv)))

            generated.extend(draft_tokens[:accepted])
            pos += accepted

            if pos < total_len:
                all_tokens = all_tokens.at[pos - 1].set(jnp.int32(draft_tokens[accepted - 1]))

        generated = generated[:max_new_tokens]
        token_ids.extend(generated)
        return self.tokenizer.decode(token_ids)
