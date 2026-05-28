import jax.numpy as jnp
import jax
from jax import lax

def f(x):
    return x**2-3

f_grad = jax.grad(f)

def newton_lax(x0,t=1e-5,max_i=100):
    def cond_fun(state):
        x,i = state
        return (jnp.abs(f(x)) > t) & (i <= max_i)

    def body_fun(state):
        x,i = state
        return (x - (f(x)/f_grad(x)),i+1)
    
    a, b = lax.while_loop(cond_fun,body_fun,(x0,1))
    return a, b

def main():
    jit_newton = jax.jit(newton_lax)
    root, iters = jit_newton(4.)
    print(f"root: {root:.6f}  iterations: {iters}")
    print(f"verify f(root) = {f(root):.8f}")

if __name__ == "__main__":
    main()
