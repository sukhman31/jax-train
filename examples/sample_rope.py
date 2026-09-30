import os, sys
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from transformer import RoPE

x = jnp.array([[
    [[1., 2., 3., 4.],
     [5., 1., 2., 0.],
     [0., 3., 1., 2.]],
]])
print(f"input shape: {x.shape}")

rope = RoPE(head_dim=4, base=100)
x_rotated = rope(x)

print("rotated (position 0):", x_rotated[0, 0, 0])
print("rotated (position 1):", x_rotated[0, 0, 1])
print("rotated (position 2):", x_rotated[0, 0, 2])

expected = jnp.array([1.0185, 0.9950, 5.2881, 0.0998])
print(f"\nexpected (hand-computed): {expected}")
print(f"matches: {jnp.allclose(x_rotated[0, 0, 1], expected, atol=1e-3)}")
