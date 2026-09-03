import jax
import jax.numpy as jnp
from flax import nnx

from .lora_param import LoRAParam

class LoRAWrapped(nnx.Module):
    def __init__(self, base_matrix: nnx.Param, rngs: nnx.Rngs, rank):
        self.base_matrix = base_matrix
        self.A = LoRAParam(
            jnp.zeros((base_matrix.shape[0],rank))
        )
        self.B = LoRAParam(
            jax.random.normal(rngs.params(),(rank,base_matrix.shape[1]))
        )
        self.rank = rank

    def __call__(self, x):
        return x @ self.base_matrix.value + (x @ self.A.value) @ self.B.value