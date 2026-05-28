import jax.numpy as jnp
import jax
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

def init_params(key):
    key1, key2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(key1,(4,2)),
        "b1": jnp.zeros(4),
        "w2": jax.random.normal(key2,(2,4)),
        "b2": jnp.zeros(2)
    }

def forward(x,params):
    h = jax.nn.relu(jnp.dot(params["w1"], x) + params["b1"])
    y = jnp.dot(params["w2"], h) + params["b2"]
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

    mesh = jax.make_mesh((4,), ('batch',))
    jax.set_mesh(mesh)

    params = init_params(param_key)
    grad_fn = jax.grad(loss, argnums=2)
    batch_grad_fn = jax.vmap(grad_fn,in_axes=(0,0,None))
    batch_loss = jax.jit(jax.vmap(loss, in_axes=(0,0,None)))
    jit_grad = jax.jit(batch_grad_fn)
    
    xs = jax.random.normal(data_key, (12,2))
    targets = jax.random.normal(data_key, (12,2))

    #existing xs shape (10, 2) — shard over batch
    xs_sharded = jax.device_put(xs, jax.P('batch', None))

    #existing targets shape (10, 2) — shard over batch  
    targets_sharded = jax.device_put(targets, jax.P('batch', None))

    # params dict — replicate on all devices
    params_sharded = jax.tree.map(
        lambda p: jax.device_put(p, jax.P(None)), 
        params
    )

    print(jax.typeof(xs_sharded))
    print(jax.typeof(params_sharded['w1']))

    for i in range(num_batches):
        loss_grad = jit_grad(xs_sharded,targets_sharded,params_sharded)
        mean_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), loss_grad)

        params_sharded = jax.tree.map(lambda p, g: p - lr * g, params_sharded, mean_grads)

        if i%10==0:
            print(jnp.mean(batch_loss(xs_sharded,targets_sharded,params_sharded)))

if __name__=="__main__":
    main()