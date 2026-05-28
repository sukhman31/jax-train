import jax
import jax.numpy as jnp
from flax import nnx
import optax

class MLP(nnx.Module):
    def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1)
        self.ln      = nnx.LayerNorm(dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs):
        x = self.linear1(x)
        x = self.dropout(x, rngs=rngs)
        x = self.ln(x)
        x = nnx.gelu(x)
        return self.linear2(x)
    
def loss_fn(model,x,target,rngs):
    preds = model(x, rngs=rngs)
    return jnp.mean((preds-target)**2)

@nnx.jit
def train_step(model,optimizer,x,target,rngs):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, target, rngs)
    optimizer.update(model,grads)
    return loss

def main():
    num_iters = 1000
    key = jax.random.key(31)

    mlp = MLP(2, 4, 2, nnx.Rngs(0))
    optimizer = nnx.Optimizer(mlp, optax.adam(1e-3), wrt=nnx.Param)

    xs = jax.random.normal(key, (32,2))
    targets = jax.random.normal(key, (32,2))
    

    for i in range(num_iters):
        rngs = nnx.Rngs(i)
        loss = train_step(mlp,optimizer,xs,targets,rngs)
        if i % 20 == 0:
            print(f"step {i:3d}  loss {loss:.4f}")

if __name__ == "__main__":
    main()