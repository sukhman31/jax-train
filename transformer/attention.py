from flax import nnx
import jax
import jax.numpy as jnp

class ScaledDotProdAttention(nnx.Module):

    def __init__(self, input_dim, output_dim, rngs: nnx.Rngs):
        self.q = nnx.Param(
            jax.random.normal(rngs.params(),(input_dim,output_dim))
        )
        self.k = nnx.Param(
            jax.random.normal(rngs.params(),(input_dim,output_dim))
        )
        self.v = nnx.Param(
            jax.random.normal(rngs.params(),(input_dim,output_dim))
        )
        self.dropout = nnx.Dropout(0.2)
    
    def __call__(self, embeddings, rngs: nnx.Rngs):
        query = embeddings @ self.q.value
        key = embeddings @ self.k.value
        value = embeddings @ self.v.value
        attention_scores = query @ jnp.swapaxes(key, -1, -2)
        mask = jnp.tril(jnp.ones((attention_scores.shape[-1],attention_scores.shape[-1]), dtype=bool))
        attention_scores = jnp.where(mask,attention_scores,-jnp.inf)
        attention_weights = nnx.softmax(attention_scores/key.shape[-1]**0.5,axis=-1)
        attention_weights = self.dropout(attention_weights, rngs=rngs)
        attention = attention_weights @ value
        return attention
    

class MultiHeadAttention(nnx.Module):
    
    def __init__(self, input_dim, output_dim, num_heads, rngs: nnx.Rngs):
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = output_dim//num_heads
        self.q = nnx.Param(
            jax.random.normal(rngs.params(),(input_dim,output_dim))
        )
        self.k = nnx.Param(
            jax.random.normal(rngs.params(),(input_dim,output_dim))
        )
        self.v = nnx.Param(
            jax.random.normal(rngs.params(),(input_dim,output_dim))
        )
        self.dropout = nnx.Dropout(0.2)
    
    def __call__(self, embeddings, rngs: nnx.Rngs):
        b, tokens, _ = embeddings.shape

        query = embeddings @ self.q.value
        key = embeddings @ self.k.value
        value = embeddings @ self.v.value

        query = jnp.reshape(query, (b,tokens,self.num_heads,self.head_dim)).swapaxes(1,2)
        key = jnp.reshape(key, (b,tokens,self.num_heads,self.head_dim)).swapaxes(1,2)
        value = jnp.reshape(value, (b,tokens,self.num_heads,self.head_dim)).swapaxes(1,2)

        attention_scores = query @ jnp.swapaxes(key, -1, -2)
        mask = jnp.tril(jnp.ones((attention_scores.shape[-1],attention_scores.shape[-1]), dtype=bool))
        attention_scores = jnp.where(mask,attention_scores,-jnp.inf)
        attention_weights = nnx.softmax(attention_scores/key.shape[-1]**0.5,axis=-1)
        attention_weights = self.dropout(attention_weights, rngs=rngs)
        attention = attention_weights @ value
        attention = jnp.swapaxes(attention, 1, 2)
        return jnp.reshape(attention, (b,tokens, self.num_heads*self.head_dim))