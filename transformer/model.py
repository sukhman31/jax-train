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

    def __call__(self, x, rngs: nnx.Rngs):
        x = self.embedding(x)
        for block in self.gpt_blocks:
            x = block(x, rngs)
        x = self.layer_norm(x)
        return self.linear(x)