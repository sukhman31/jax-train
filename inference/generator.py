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

    def generate(self, prompt, max_new_tokens=100, temperature=0.8):
        token_ids = self.tokenizer.encode(prompt)
        max_seq_len = self.config['max_seq_len']

        for _ in range(max_new_tokens):
            context = token_ids[-max_seq_len:]
            x = jnp.array([context])
            logits = self.model(x, self.rngs)
            next_logits = logits[0, -1, :]

            if temperature <= 0:
                next_id = int(jnp.argmax(next_logits))
            else:
                next_id = int(jax.random.categorical(self.rngs.sample(), next_logits / temperature))

            token_ids.append(next_id)

        return self.tokenizer.decode(token_ids)
