from flax import nnx

from .attention import MultiHeadAttention
from .layer_norm import LayerNorm
from .feed_forward import FeedForward


class GPTBlock(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.norm1 = LayerNorm(config['eps'], config['embed_dim'])
        self.attention = MultiHeadAttention(config['input_dim'], config['output_dim'], config['num_heads'], rngs)
        self.norm2 = LayerNorm(config['eps'], config['output_dim'])
        self.feed_forward = FeedForward(config['output_dim'], config['scaling_factor']*config['output_dim'], False, rngs)
        self.dropout =  nnx.Dropout(config['dropout'])
    
    def __call__(self, x, rngs: nnx.Rngs):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x, rngs)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x,rngs=rngs)

        return x + shortcut