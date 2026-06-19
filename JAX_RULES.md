# JAX Rules

A living reference. Add to this as new rules are discovered while porting batran-lib.

---

## 1. The two-phase mental model

Every jitted function runs in two phases:

```
call 1:  TRACE (Python executes) → COMPILE (XLA takes over)
call 2+: COMPILED VERSION RUNS   (Python never executes again)
```

Every rule below is a consequence of this.

---

## 2. Purity — the one rule they all derive from

Functions passed to `jit`, `grad`, `vmap` must be **pure**:
- Same inputs → same output, always
- No mutation of inputs
- No reading/writing global state
- No side effects

---

## 3. Control flow — the most common gotcha

Ask yourself: does this `if / while / for` depend on an **array value** or a **shape/Python scalar**?

| Condition depends on | Python control flow | JAX equivalent |
|---|---|---|
| Array value | ❌ | `jnp.where`, `lax.cond`, `lax.while_loop` |
| Shape / dtype | ✅ | plain Python fine |
| Python int/bool | ✅ | plain Python fine |

```python
# ❌ fails — value unknown at trace time
if x[0] > 0:
    return x * 2

# ✅ use jnp.where for element-wise branching
return jnp.where(x > 0, x * 2, x)

# ✅ shape is known at trace time
if x.shape[0] > 4:
    return x[:4]
```

### lax control flow equivalents

```python
lax.while_loop(cond_fn, body_fn, init)   # while loop on dynamic condition
lax.cond(condition, true_fn, false_fn)   # if/else on array value
lax.fori_loop(start, stop, body_fn)      # for loop with dynamic bounds
lax.scan(fn, init, xs)                   # for loop accumulating state (prefer over fori_loop)
```

---

## 4. grad rules

- Output must be a **scalar**
- Input being differentiated must be **float arrays** (or pytree of floats) — no integers
- Use `argnums` to specify which argument to differentiate w.r.t.
- Returned gradient has the **same pytree structure** as the differentiated argument

---

## 5. vmap rules

- `in_axes` must match the number of arguments
- `None` = argument is shared across batch (not mapped)
- All mapped arguments must have the same size on the mapped axis
- Compose with `jit` — put `jit` on the outside: `jax.jit(jax.vmap(fn))`

---

## 6. jit rules

- First call is slow (trace + compile), subsequent calls are fast
- Array **shapes must be static** — different shapes trigger recompilation
- `print` inside a jitted function only runs during tracing (call 1), never again
- Use `jax.debug.print` if you need a print that runs every call

---

## 7. Random keys — always explicit

```python
key = jax.random.key(seed)
key1, key2 = jax.random.split(key)   # split before every use
```

- Never reuse a key — split it
- Pass keys explicitly as function arguments, never store them in global state

---

## 8. Pytrees

- JAX treats dicts, lists, tuples as **nodes** and arrays as **leaves**
- `jax.tree.map(fn, tree)` applies `fn` to every leaf
- Two pytrees passed to `tree.map` must have **identical structure**
- Custom classes must be registered with `@jax.tree_util.register_pytree_node_class`
  - `tree_flatten` → split into (leaves, aux): leaves are arrays, aux is non-array metadata
  - `tree_unflatten` → reconstruct from (aux, leaves)

---

## 9. Integers will silently break grad

```python
# ❌ integer array — grad returns zeros, no error
params = {"w": jnp.array([1, 2, 3])}

# ✅ float array
params = {"w": jnp.array([1.0, 2.0, 3.0])}
```

Always initialise parameters with `jnp.zeros`, `jnp.ones`, or `jax.random.normal` — these default to float32.

---

## 10. No in-place mutation

```python
# ❌ JAX arrays don't support item assignment
params["w"][0] = 1.0

# ✅ return a new array
params = {**params, "w": params["w"].at[0].set(1.0)}
```

The `.at[].set()` / `.at[].add()` syntax is JAX's immutable update API.

---

## 11. Composition order

```
jit(vmap(grad(fn)))   ✅ standard order
```
- `grad` innermost — transforms the function
- `vmap` middle — vectorizes it
- `jit` outermost — compiles the whole thing

---

## Discovered during batran-lib port

> Add rules here as new issues are encountered.
