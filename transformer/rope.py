from flax import nnx
import jax
import jax.numpy as jnp

class RoPE(nnx.Module):
    def __init__(self, head_dim, base=10000):
        self.head_dim = head_dim
        self.pairs = head_dim//2
        self.inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2) / head_dim))

    def __call__(self, x):
        seq_len = x.shape[-2]
        positions = jnp.arange(seq_len)
        angles = jnp.outer(positions, self.inv_freq)

        cos_vals = jnp.cos(angles)
        sin_vals = jnp.sin(angles)

        cos_full = jnp.concatenate([cos_vals, cos_vals], axis=-1)
        sin_full = jnp.concatenate([sin_vals, sin_vals], axis=-1)

        x0, x1 = x[..., :self.head_dim//2], x[..., self.head_dim//2:]
        half_rotate = jnp.concatenate([-x1, x0], axis=-1)

        return x * cos_full + half_rotate * sin_full
