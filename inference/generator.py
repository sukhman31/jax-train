import pickle

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
        num_heads = self.config['num_heads']
        head_dim = self.config['output_dim'] // num_heads
        cache = []
        for _ in range(num_blocks):
            k = jnp.zeros((batch_size, num_heads, max_len, head_dim))
            v = jnp.zeros((batch_size, num_heads, max_len, head_dim))
            cache.append((k, v))
        return cache

    def generate(self, prompt, max_new_tokens=100, temperature=0.8):
        token_ids = self.tokenizer.encode(prompt)
        prompt_len = len(token_ids)
        max_len = prompt_len + max_new_tokens

        cache = self._init_cache(1, max_len)

        x = jnp.array([token_ids])
        logits, cache = self.model(x, self.rngs, cache=cache, cache_pos=0)
        next_logits = logits[0, -1, :]

        for i in range(max_new_tokens):
            if temperature <= 0:
                next_id = int(jnp.argmax(next_logits))
            else:
                next_id = int(jax.random.categorical(self.rngs.sample(), next_logits / temperature))

            token_ids.append(next_id)

            x = jnp.array([[next_id]])
            logits, cache = self.model(x, self.rngs, cache=cache, cache_pos=prompt_len + i)
            next_logits = logits[0, -1, :]

        return self.tokenizer.decode(token_ids)
