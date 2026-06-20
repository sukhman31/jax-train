from flax import nnx
import jax
import jax.numpy as jnp

# TODO: implement static positional encoding instead of learned encoding

class Embedding(nnx.Module):
    def __init__(self,vocab_size,dimensions,max_seq_len,rngs: nnx.Rngs):
        self.embed = nnx.Param(
            jax.random.normal(rngs.params(),(vocab_size,dimensions))*0.02
        )
        self.positional = nnx.Param(
            jax.random.normal(rngs.params(),(max_seq_len,dimensions))*0.02
        )

    def __call__(self,token_ids_batch):
        embeddings = self.embed[token_ids_batch]
        positions = token_ids_batch.shape[1]
        positional_encodings = self.positional[:positions] 
        print(positional_encodings.shape)
        return embeddings + positional_encodings