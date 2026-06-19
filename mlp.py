import jax
import jax.numpy as jnp

def init_params(key):
    key1, key2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(key1,(4,2)),
        "b1": jnp.zeros(4),
        "w2": jax.random.normal(key2,(2,4)),
        "b2": jnp.zeros(2)
    }

def forward(x, params):
    h = jax.nn.relu(jnp.dot(params["w1"],x)+params["b1"])
    y = jnp.dot(params["w2"],h)+params["b2"]
    return y  

def loss(x,target,params):
    pred = forward(x,params)
    loss = jnp.mean((pred-target)**2)
    return loss

def main():
    num_batches = 1000
    lr = 0.01

    key = jax.random.key(31)
    data_key, param_key = jax.random.split(key)

    params = init_params(param_key)
    grad_fn = jax.grad(loss, argnums=2)
    batch_grad_fn = jax.vmap(grad_fn,in_axes=(0,0,None))
    batch_loss = jax.jit(jax.vmap(loss, in_axes=(0,0,None)))
    jit_grad = jax.jit(batch_grad_fn)
    
    xs = jax.random.normal(data_key, (10,2))
    targets = jax.random.normal(data_key, (10,2))

    for i in range(num_batches):
        #vmap over grad_fn means each param gets a gradient per sample so it will be like (n, shape_0, shape_1) and then we have to averge
        #over n
        loss_grad = jit_grad(xs,targets,params)
        mean_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), loss_grad)

        params = jax.tree.map(lambda p, g: p - lr * g, params, mean_grads)

        if i%10==0:
            print(jnp.mean(batch_loss(xs,targets,params)))

if __name__ == "__main__":
    main()
