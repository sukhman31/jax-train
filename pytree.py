import jax.numpy as jnp
import jax

@jax.tree_util.register_pytree_node_class
class Parameter:
    def __init__(self,values,name,frozen):
        self.values = values
        self.name = name
        self.frozen = frozen

    def tree_flatten(self):
        leaves = (self.values,)
        aux = (self.name, self.frozen)
        return leaves, aux
    
    @classmethod
    def tree_unflatten(cls,aux,leaves):
        (values,) = leaves
        name, frozen = aux
        return cls(values,name,frozen)
    
    def __repr__(self):
        return f"Parameter(values={self.values}, name={self.name}, frozen={self.frozen})"

def loss(parameter):
    return jnp.mean(parameter.values**2)

def main():
    key = jax.random.key(31)
    #created this as a list because it gets counted as a node and when iterating
    #in map lambda each one gets as separate leaf as opposed to it getting treated
    #as one leaf had i created one simply as (4,3) leaf
    xs = [jax.random.normal(key, (4,)) for _ in range(3)]
    parameters = jax.tree.map(lambda x: Parameter(x,"weights",False), xs)
    grad_fn = jax.grad(loss)
    jit_grad = jax.jit(grad_fn)

    #here we dont need vamp because we are not slicing aross axes, its three separate
    #objects in a list vmap would be needed if we had just one (4,3) leaf
    for p in parameters:
        grad = jit_grad(p)
        print(grad)



if __name__ == "__main__":
    main()
