import os
import pickle
import time

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
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        sample = raw_text[:100_000]
        self.tokenizer.train(sample)
        print(f"tokenizer trained on {len(sample)} chars, {len(self.tokenizer.pairs_dict)} merges")
        token_ids = self.tokenizer.encode(raw_text)

        self.sampler = SlidingWindowSampler(
            seq_len=self.config['seq_len'],
            batch_size=self.config['batch_size'],
            token_ids=token_ids,
            stride=self.config.get('stride', 1),
        )
        print(f"corpus: {len(raw_text)} chars -> {len(token_ids)} tokens, "
              f"{self.sampler.num_batches} batches/epoch")

    def train(self, file_path, num_epochs):
        self.prepare_data(file_path)
        total_batches = self.sampler.num_batches
        train_start = time.time()

        for i in range(num_epochs):
            self.sampler.shuffle()
            losses = []
            epoch_start = time.time()
            for step, (x, y) in enumerate(self.sampler):
                losses.append(self.train_step(x, y))
                if step % 50 == 0:
                    print(f"  epoch {i+1}/{num_epochs} | batch {step+1}/{total_batches} | loss: {float(losses[-1]):.4f}", flush=True)
            avg_loss = jnp.mean(jnp.array(losses))
            elapsed = time.time() - epoch_start
            total = time.time() - train_start
            print(f"epoch {i+1}/{num_epochs} | avg loss: {float(avg_loss):.4f} | {elapsed:.1f}s | total: {total:.0f}s", flush=True)

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
