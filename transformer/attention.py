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

    def __init__(self, input_dim, output_dim, num_heads, rngs: nnx.Rngs, num_kv_heads=None):
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = output_dim//num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_groups = num_heads // self.num_kv_heads
        self.q = Projection(input_dim, output_dim, rngs)
        self.k = Projection(input_dim, self.num_kv_heads * self.head_dim, rngs)
        self.v = Projection(input_dim, self.num_kv_heads * self.head_dim, rngs)
        self.rope = RoPE(self.head_dim)
        self.dropout = nnx.Dropout(0.2)

    def __call__(self, embeddings, rngs: nnx.Rngs, cache=None, cache_pos=None):
        b, tokens, _ = embeddings.shape

        query = self.q(embeddings)
        key = self.k(embeddings)
        value = self.v(embeddings)

        query = jnp.reshape(query, (b,tokens,self.num_heads,self.head_dim)).swapaxes(1,2)
        key = jnp.reshape(key, (b,tokens,self.num_kv_heads,self.head_dim)).swapaxes(1,2)
        value = jnp.reshape(value, (b,tokens,self.num_kv_heads,self.head_dim)).swapaxes(1,2)

        offset = 0 if cache_pos is None else cache_pos
        query = self.rope(query, offset)
        key = self.rope(key, offset)

        if cache is not None:
            cache_k, cache_v = cache
            indices = jnp.arange(tokens) + cache_pos
            cache_k = cache_k.at[:, :, indices, :].set(key)
            cache_v = cache_v.at[:, :, indices, :].set(value)
            key = cache_k
            value = cache_v
            new_cache = (cache_k, cache_v)
            valid_len = cache_pos + tokens
        else:
            new_cache = None
            valid_len = tokens

        if self.num_groups > 1:
            key = jnp.repeat(key, self.num_groups, axis=1)
            value = jnp.repeat(value, self.num_groups, axis=1)

        attention_scores = query @ jnp.swapaxes(key, -1, -2)
        total_len = key.shape[-2]
        row_ids = jnp.arange(tokens) + offset
        col_ids = jnp.arange(total_len)
        mask = (col_ids[None, :] <= row_ids[:, None]) & (col_ids[None, :] < valid_len)
        attention_scores = jnp.where(mask,attention_scores,-jnp.inf)
        attention_weights = nnx.softmax(attention_scores/self.head_dim**0.5,axis=-1)
        attention_weights = self.dropout(attention_weights, rngs=rngs)
        attention = attention_weights @ value
        attention = jnp.swapaxes(attention, 1, 2)
        return jnp.reshape(attention, (b,tokens, self.num_heads*self.head_dim)), new_cache