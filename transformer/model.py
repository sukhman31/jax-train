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

    def __call__(self, x, rngs: nnx.Rngs, cache=None, cache_pos=None, exit_after=None):
        x = self.embedding(x)
        num_blocks = len(self.gpt_blocks) if exit_after is None else exit_after
        new_caches = []
        total_aux_loss = 0.0
        for i in range(num_blocks):
            block_cache = cache[i] if cache is not None else None
            x, new_cache, aux_loss = self.gpt_blocks[i](x, rngs, block_cache, cache_pos)
            new_caches.append(new_cache)
            total_aux_loss = total_aux_loss + aux_loss
        x = self.layer_norm(x)
        return self.linear(x), new_caches, total_aux_loss