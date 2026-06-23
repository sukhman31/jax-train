from flax import nnx
import jax
import jax.numpy as jnp

class LayerNorm(nnx.Module):
    def __init__(self,eps,embed_dim):
        self.eps = eps
        self.scale = nnx.Param(
            jnp.ones(embed_dim)
        )
        self.shift = nnx.Param(
            jnp.zeros(embed_dim)
        )
    
    def __call__(self, embeddings):
        mean = jnp.mean(embeddings, axis=-1, keepdims=True)
        var = jnp.var(embeddings, axis=-1, keepdims=True)
        norm = (embeddings-mean)/jnp.sqrt(var+self.eps)
        return self.scale.value*norm + self.shift.value