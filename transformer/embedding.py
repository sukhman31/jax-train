from flax import nnx
import jax
import jax.numpy as jnp

class Embedding(nnx.Module):
    def __init__(self, vocab_size, dimensions, rngs: nnx.Rngs):
        self.embed = nnx.Param(
            jax.random.normal(rngs.params(),(vocab_size,dimensions))*0.02
        )

    def __call__(self, token_ids_batch):
        embeddings = self.embed[token_ids_batch]
        return embeddings