from flax import nnx
import jax
import jax.numpy as jnp

class FeedForward(nnx.Module):
    def __init__(self, embed_dim, scale_dim, use_bias, rngs: nnx.Rngs):
        self.gate = nnx.Linear(embed_dim, scale_dim, use_bias=use_bias, rngs=rngs)
        self.up = nnx.Linear(embed_dim, scale_dim, use_bias=use_bias, rngs=rngs)
        self.down = nnx.Linear(scale_dim, embed_dim, use_bias=use_bias, rngs=rngs)

    def __call__(self, embeddings):
        return self.down(jax.nn.silu(self.gate(embeddings)) * self.up(embeddings))