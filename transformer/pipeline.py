from flax import nnx
import jax
import jax.numpy as jnp

from embedding import Embedding
from attention import ScaledDotProdAttention

# --- config ---
VOCAB_SIZE  = 1000
DIMS        = 64
ATTENTION_DIMS = 32
MAX_SEQ_LEN = 256
RNGS        = nnx.Rngs(0)

# --- components ---
embedding = Embedding(VOCAB_SIZE, DIMS, MAX_SEQ_LEN, RNGS)
attention = ScaledDotProdAttention(DIMS, ATTENTION_DIMS, RNGS)

# --- dummy input ---
token_ids = jnp.array([[3, 14, 56], [4, 42, 7]])  # (batch=2, seq_len=3)

# --- forward pass ---
x = embedding(token_ids)
print(f"after embedding:  {x.shape}")   # (2, 3, 64)

x = attention(x)
print(f"after attention:  {x.shape}")   # (2, 3, 32)
