from flax import nnx
import jax
import jax.numpy as jnp

class FeedForward(nnx.Module):
    def __init__(self, embed_dim, scale_dim, use_bias, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(embed_dim,scale_dim,use_bias=use_bias,rngs=rngs)
        self.linear2 = nnx.Linear(scale_dim,embed_dim,use_bias=use_bias,rngs=rngs)
    
    def __call__(self,embeddings):
        x = self.linear1(embeddings)
        x = nnx.gelu(x)
        return self.linear2(x)