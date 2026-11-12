import os
import pickle

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from transformer import Model
from tokenizers.bpe_tokenizer import BPETokenizer
from sampler import SlidingWindowSampler


class Trainer(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.config = config
        self.model = Model(config, rngs)
        self.optimizer = nnx.Optimizer(self.model, optax.adam(config['learning_rate']), wrt=nnx.Param)
        self.rngs = rngs
        self.tokenizer = BPETokenizer(max_merges=config['max_merges'])
        self.sampler = None

    @staticmethod
    def loss_fn(model, x, target, rngs):
        preds, _ = model(x, rngs)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(preds, target))

    @nnx.jit
    def train_step(self, x, target):
        loss, grads = nnx.value_and_grad(Trainer.loss_fn)(self.model, x, target, self.rngs)
        self.optimizer.update(self.model, grads)
        return loss

    def prepare_data(self, file_path):
        self.tokenizer.ingest_docs(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        token_ids = self.tokenizer.encode(raw_text)

        self.sampler = SlidingWindowSampler(
            seq_len=self.config['seq_len'],
            batch_size=self.config['batch_size'],
            token_ids=token_ids,
        )
        print(f"corpus: {len(raw_text)} chars -> {len(token_ids)} tokens, "
              f"{self.sampler.num_batches} batches/epoch")

    def train(self, file_path, num_epochs):
        self.prepare_data(file_path)

        for i in range(num_epochs):
            self.sampler.shuffle()
            losses = []
            for x, y in self.sampler:
                losses.append(self.train_step(x, y))
            if i % 5 == 0:
                avg_loss = jnp.mean(jnp.array(losses))
                print(f'Loss after {i+1} epochs: {avg_loss:.2f}')

    def save(self, path):
        _, model_state = nnx.split(self.model)
        checkpoint = {
            'config': self.config,
            'model_state': model_state,
            'tokenizer_pairs': self.tokenizer.pairs_dict,
            'tokenizer_vocab': self.tokenizer.vocab,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"saved checkpoint to {path}")
