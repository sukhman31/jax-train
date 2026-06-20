from flax import nnx
import jax
import jax.numpy as jnp

class ScaledDotProdAttention(nnx.Module):

    def __init__(self,input_dim,output_dim,rngs: nnx.Rngs):
        self.q = nnx.Param(
            jax.random.normal(rngs.params(),(input_dim,output_dim))
        )
        self.k = nnx.Param(
            jax.random.normal(rngs.params(),(input_dim,output_dim))
        )
        self.v = nnx.Param(
            jax.random.normal(rngs.params(),(input_dim,output_dim))
        )
    
    def __call__(self,embeddings):
        query = embeddings @ self.q.value
        key = embeddings @ self.k.value
        value = embeddings @ self.v.value
        attention_scores = query @ jnp.swapaxes(key, -1, -2)
        mask = jnp.tril(jnp.ones((attention_scores.shape[-1],attention_scores.shape[-1]), dtype=bool))
        attention_scores = jnp.where(mask,attention_scores,-jnp.inf)
        attention_weights = nnx.softmax(attention_scores/key.shape[-1]**0.5,axis=-1)
        attention = attention_weights @ value
        return attention