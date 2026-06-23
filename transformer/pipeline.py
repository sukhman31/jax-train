from flax import nnx
import jax
import jax.numpy as jnp

from embedding import Embedding
from attention import ScaledDotProdAttention, MultiHeadAttention
from layer_norm import LayerNorm

# --- config ---
VOCAB_SIZE  = 1000
DIMS        = 64
ATTENTION_DIMS = 32
HEADS = 4
MAX_SEQ_LEN = 256
RNGS        = nnx.Rngs(0)

# --- components ---
embedding = Embedding(VOCAB_SIZE, DIMS, MAX_SEQ_LEN, RNGS)
attention = ScaledDotProdAttention(DIMS, ATTENTION_DIMS, RNGS)
mha = MultiHeadAttention(DIMS, ATTENTION_DIMS, HEADS, RNGS)
layer_norm = LayerNorm(1e-5, ATTENTION_DIMS)

# --- dummy input ---
token_ids = jnp.array([[3, 14, 56], [4, 42, 7]])  # (batch=2, seq_len=3)

# --- forward pass ---
x = embedding(token_ids)
print(f"after embedding:  {x.shape}")   # (2, 3, 64)

y = attention(x, RNGS)
print(f"after attention:  {y.shape}")   # (2, 3, 32)

z = mha(x, RNGS)
print(f"after MHA:        {z.shape}")   # (2, 3, 32)

n = layer_norm(z)
print(f"after LayerNorm:  {n.shape}")   # (2, 3, 32)
