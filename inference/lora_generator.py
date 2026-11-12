import pickle

import jax
import jax.numpy as jnp
from flax import nnx

from transformer import Model
from tokenizers.bpe_tokenizer import BPETokenizer
from lora.lora_wrapped import LoRAWrapped


class LoRAGenerator(nnx.Module):
    def __init__(self, base_checkpoint_path, adapter_path, rngs: nnx.Rngs):
        with open(base_checkpoint_path, 'rb') as f:
            base_checkpoint = pickle.load(f)
        with open(adapter_path, 'rb') as f:
            adapter = pickle.load(f)

        self.config = base_checkpoint['config']
        self.rngs = rngs

        self.tokenizer = BPETokenizer(max_merges=self.config['max_merges'])
        self.tokenizer.pairs_dict = base_checkpoint['tokenizer_pairs']
        self.tokenizer.vocab = base_checkpoint['tokenizer_vocab']

        self.model = Model(self.config, rngs)
        nnx.update(self.model, base_checkpoint['model_state'])

        rank = adapter['rank']
        targets = adapter['targets']
        for block in self.model.gpt_blocks:
            if 'q' in targets:
                block.attention.q = LoRAWrapped(block.attention.q.weight, rngs, rank)
            if 'v' in targets:
                block.attention.v = LoRAWrapped(block.attention.v.weight, rngs, rank)

        nnx.update(self.model, adapter['lora_state'])
        self.model.eval()

    def generate(self, prompt, max_new_tokens=100, temperature=0.8):
        token_ids = self.tokenizer.encode(prompt)

        x = jnp.array([token_ids])
        logits, cache = self.model(x, self.rngs, cache=None)
        next_logits = logits[0, -1, :]

        for _ in range(max_new_tokens):
            if temperature <= 0:
                next_id = int(jnp.argmax(next_logits))
            else:
                next_id = int(jax.random.categorical(self.rngs.sample(), next_logits / temperature))

            token_ids.append(next_id)

            x = jnp.array([[next_id]])
            logits, cache = self.model(x, self.rngs, cache=cache)
            next_logits = logits[0, -1, :]

        return self.tokenizer.decode(token_ids)
