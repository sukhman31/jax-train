import flax.nnx as nnx
import jax

class Projection(nnx.Module):
    def __init__(self,input_dim,output_dim, rngs: nnx.Rngs):
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(),(input_dim,output_dim))
        )

    def __call__(self, x):
        return x @ self.weight.value