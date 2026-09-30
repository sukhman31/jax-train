from flax import nnx
import jax
import jax.numpy as jnp

from .projection import Projection
from .rope import RoPE

class ScaledDotProdAttention(nnx.Module):

    def __init__(self, input_dim, output_dim, rngs: nnx.Rngs):
        self.q = Projection(input_dim,output_dim,rngs)
        self.k = Projection(input_dim,output_dim,rngs)
        self.v = Projection(input_dim,output_dim,rngs)
        self.dropout = nnx.Dropout(0.2)
    
    def __call__(self, embeddings, rngs: nnx.Rngs):
        query = self.q(embeddings)
        key = self.k(embeddings)
        value = self.v(embeddings)
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
        self.q = Projection(input_dim,output_dim,rngs)
        self.k = Projection(input_dim,output_dim,rngs)
        self.v = Projection(input_dim,output_dim,rngs)
        self.rope = RoPE(self.head_dim)
        self.dropout = nnx.Dropout(0.2)

    def __call__(self, embeddings, rngs: nnx.Rngs):
        b, tokens, _ = embeddings.shape

        query = self.q(embeddings)
        key = self.k(embeddings)
        value = self.v(embeddings)

        query = jnp.reshape(query, (b,tokens,self.num_heads,self.head_dim)).swapaxes(1,2)
        key = jnp.reshape(key, (b,tokens,self.num_heads,self.head_dim)).swapaxes(1,2)
        value = jnp.reshape(value, (b,tokens,self.num_heads,self.head_dim)).swapaxes(1,2)

        query = self.rope(query)
        key = self.rope(key)

        attention_scores = query @ jnp.swapaxes(key, -1, -2)
        mask = jnp.tril(jnp.ones((attention_scores.shape[-1],attention_scores.shape[-1]), dtype=bool))
        attention_scores = jnp.where(mask,attention_scores,-jnp.inf)
        attention_weights = nnx.softmax(attention_scores/key.shape[-1]**0.5,axis=-1)
        attention_weights = self.dropout(attention_weights, rngs=rngs)
        attention = attention_weights @ value
        attention = jnp.swapaxes(attention, 1, 2)
        return jnp.reshape(attention, (b,tokens, self.num_heads*self.head_dim))