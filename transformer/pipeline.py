from flax import nnx
import jax
import jax.numpy as jnp
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tokenizers.bpe_tokenizer import BPETokenizer
from .embedding import Embedding
from .attention import ScaledDotProdAttention, MultiHeadAttention
from .layer_norm import LayerNorm
from .feed_forward import FeedForward
from .gpt_block import GPTBlock
from .model import Model

# --- config ---
CONFIG = {
    'vocab_size':    1000,
    'embed_dim':     64,
    'max_seq_len':   256,
    'num_heads':     4,
    'input_dim':     64,
    'output_dim':    64,
    'scaling_factor': 4,
    'dropout':       0.1,
    'eps':           1e-5,
    'num_of_blocks': 2,
}
RNGS = nnx.Rngs(0)

# --- tokenizer ---
tokenizer = BPETokenizer(max_merges=744)
shakespeare_path = os.path.join(os.path.dirname(__file__), '..', 'resources', 'shakespeare_sample.txt')
tokenizer.ingest_docs(shakespeare_path)
encoded = tokenizer.encode("To be or not to be")
decoded = tokenizer.decode(encoded)
print(f"encoded: {encoded}")
print(f"decoded: {decoded}")

# --- dummy input ---
token_ids = jnp.array([encoded[:3], encoded[:3]])  # (batch=2, seq_len=3)

# --- component checks ---
embedding  = Embedding(CONFIG['vocab_size'], CONFIG['embed_dim'], CONFIG['max_seq_len'], RNGS)
attention  = ScaledDotProdAttention(CONFIG['embed_dim'], CONFIG['output_dim'], RNGS)
mha        = MultiHeadAttention(CONFIG['embed_dim'], CONFIG['output_dim'], CONFIG['num_heads'], RNGS)
layer_norm = LayerNorm(CONFIG['eps'], CONFIG['embed_dim'])
ff         = FeedForward(CONFIG['embed_dim'], CONFIG['embed_dim'] * CONFIG['scaling_factor'], use_bias=False, rngs=RNGS)
block      = GPTBlock(CONFIG, RNGS)

x = embedding(token_ids)
print(f"after embedding:   {x.shape}")   # (2, 3, 64)

y = attention(x, RNGS)
print(f"after attention:   {y.shape}")   # (2, 3, 64)

z = mha(x, RNGS)
print(f"after MHA:         {z.shape}")   # (2, 3, 64)

n = layer_norm(z)
print(f"after LayerNorm:   {n.shape}")   # (2, 3, 64)

f = ff(n)
print(f"after FeedForward: {f.shape}")   # (2, 3, 64)

b = block(x, RNGS)
print(f"after GPTBlock:    {b.shape}")   # (2, 3, 64)

# --- full model ---
model  = Model(CONFIG, RNGS)
logits = model(token_ids, RNGS)
print(f"after Model:       {logits.shape}")  # (2, 3, 1000)
