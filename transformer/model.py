from flax import nnx

from .embedding import Embedding
from .gpt_block import GPTBlock
from .rms_norm import RMSNorm

class Model(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.embedding = Embedding(config['vocab_size'], config['embed_dim'], rngs)
        self.gpt_blocks = nnx.List([GPTBlock(config, rngs) for _ in range(config['num_of_blocks'])])
        self.layer_norm = RMSNorm(config['eps'], config['embed_dim'])
        self.linear = nnx.Linear(config['embed_dim'], config['vocab_size'], rngs=rngs)

    def __call__(self, x, rngs: nnx.Rngs, cache=None):
        x = self.embedding(x)
        new_caches = []
        for i, block in enumerate(self.gpt_blocks):
            block_cache = cache[i] if cache is not None else None
            x, new_cache = block(x, rngs, block_cache)
            new_caches.append(new_cache)
        x = self.layer_norm(x)
        return self.linear(x), new_caches