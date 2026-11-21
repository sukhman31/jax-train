import jax
import jax.numpy as jnp


def sample_token(logits, generated, num_generated, key,
                 temperature, top_k, top_p, repetition_penalty):
    logits = logits.astype(jnp.float32)

    valid = (jnp.arange(generated.shape[0]) < num_generated).astype(jnp.float32)
    penalty_mask = jnp.zeros_like(logits).at[generated].add(valid)
    penalty_mask = (penalty_mask > 0).astype(jnp.float32)
    logits = jnp.where(
        penalty_mask > 0,
        jnp.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty),
        logits,
    )

    logits = logits / temperature

    sorted_indices = jnp.argsort(-logits)
    sorted_logits = logits[sorted_indices]

    topk_mask = jnp.arange(logits.shape[0]) < top_k
    sorted_logits = jnp.where(topk_mask, sorted_logits, -jnp.inf)

    sorted_probs = jax.nn.softmax(sorted_logits)
    cumulative_probs = jnp.cumsum(sorted_probs)
    nucleus_mask = cumulative_probs - sorted_probs < top_p
    sorted_logits = jnp.where(nucleus_mask, sorted_logits, -jnp.inf)

    probs = jax.nn.softmax(sorted_logits)
    sampled_index = jax.random.categorical(key, jnp.log(probs + 1e-10))
    return sorted_indices[sampled_index].astype(jnp.int32)
