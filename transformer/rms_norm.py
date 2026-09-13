from flax import nnx
import jax
import jax.numpy as jnp

class RMSNorm(nnx.Module):
    def __init__(self,eps,embed_dim):
        self.eps = eps
        self.scale = nnx.Param(
            jnp.ones(embed_dim)
        )
    
    def __call__(self, embeddings):
        rms = jnp.sqrt(jnp.mean(embeddings**2, axis=-1, keepdims=True) + self.eps)
        norm = embeddings/rms
        return self.scale.value*norm