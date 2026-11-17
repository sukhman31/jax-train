from flax import nnx

from .attention import MultiHeadAttention
from .rms_norm import RMSNorm
from .feed_forward import FeedForward
from .moe import MoELayer


class GPTBlock(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.norm1 = RMSNorm(config['eps'], config['embed_dim'])
        self.attention = MultiHeadAttention(config['input_dim'], config['output_dim'], config['num_heads'], rngs, num_kv_heads=config.get('num_kv_heads'))
        self.norm2 = RMSNorm(config['eps'], config['output_dim'])
        self.use_moe = config.get('num_experts', 0) > 0
        if self.use_moe:
            self.feed_forward = MoELayer(
                config['output_dim'],
                config['scaling_factor'] * config['output_dim'],
                config['num_experts'],
                config.get('top_k_experts', 2),
                rngs,
            )
        else:
            self.feed_forward = FeedForward(config['output_dim'], config['scaling_factor']*config['output_dim'], False, rngs)
        self.dropout = nnx.Dropout(config['dropout'])

    def __call__(self, x, rngs: nnx.Rngs, cache=None, cache_pos=None):
        shortcut = x
        x = self.norm1(x)
        x, new_cache = self.attention(x, rngs, cache, cache_pos)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        if self.use_moe:
            x, aux_loss = self.feed_forward(x)
        else:
            x, aux_loss = self.feed_forward(x), 0.0
        x = self.dropout(x, rngs=rngs)

        return x + shortcut, new_cache, aux_loss