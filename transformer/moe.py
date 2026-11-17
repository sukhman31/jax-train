import jax
import jax.numpy as jnp
from flax import nnx

from .feed_forward import FeedForward


class MoELayer(nnx.Module):
    def __init__(self, embed_dim, scale_dim, num_experts, top_k, rngs: nnx.Rngs):
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nnx.Linear(embed_dim, num_experts, use_bias=False, rngs=rngs)
        self.experts = nnx.List([
            FeedForward(embed_dim, scale_dim, False, rngs)
            for _ in range(num_experts)
        ])

    def __call__(self, x):
        b, seq_len, dim = x.shape
        flat = x.reshape(-1, dim)
        num_tokens = flat.shape[0]

        router_logits = self.router(flat)
        router_probs = jax.nn.softmax(router_logits)

        top_k_indices = jnp.argsort(-router_probs, axis=-1)[:, :self.top_k]
        top_k_probs = jnp.take_along_axis(router_probs, top_k_indices, axis=-1)
        top_k_weights = top_k_probs / jnp.sum(top_k_probs, axis=-1, keepdims=True)

        expert_outputs = jnp.stack([expert(flat) for expert in self.experts], axis=1)

        gather_indices = top_k_indices[:, :, None].repeat(dim, axis=-1)
        selected = jnp.take_along_axis(expert_outputs, gather_indices, axis=1)

        output = jnp.sum(selected * top_k_weights[:, :, None], axis=1)

        f = jnp.zeros(self.num_experts)
        for k in range(self.top_k):
            f = f.at[top_k_indices[:, k]].add(jnp.ones(num_tokens) / num_tokens)
        p = jnp.mean(router_probs, axis=0)
        aux_loss = self.num_experts * jnp.sum(f * p)

        return output.reshape(b, seq_len, dim), aux_loss
