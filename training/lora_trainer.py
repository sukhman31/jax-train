import pickle
import os

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from transformer import Model
from tokenizers.bpe_tokenizer import BPETokenizer
from sampler import SlidingWindowSampler
from lora.lora_wrapped import LoRAWrapped
from lora.lora_param import LoRAParam

class LoRATrainer(nnx.Module):
    def __init__(self, checkpoint_path, rngs: nnx.Rngs, rank, targets=('q', 'v')):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.config = checkpoint['config']
        self.rngs = rngs
        self.sampler = None
        self.rank = rank
        self.targets = targets

        self.tokenizer = BPETokenizer(max_merges=self.config['max_merges'])
        self.tokenizer.pairs_dict = checkpoint['tokenizer_pairs']
        self.tokenizer.vocab = checkpoint['tokenizer_vocab']

        self.model = Model(self.config, rngs)
        nnx.update(self.model, checkpoint['model_state'])

        for block in self.model.gpt_blocks:
            if 'q' in targets:
                block.attention.q = LoRAWrapped(block.attention.q.weight, rngs, rank)
            if 'v' in targets:
                block.attention.v = LoRAWrapped(block.attention.v.weight, rngs, rank)

        self.optimizer = nnx.Optimizer(self.model, optax.adam(self.config['learning_rate']), wrt=LoRAParam)
        
    @staticmethod
    def loss_fn(model, x, target, rngs):
        preds = model(x, rngs)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(preds, target))

    @nnx.jit
    def train_step(self, x, target):
        loss, grads = nnx.value_and_grad(LoRATrainer.loss_fn, argnums=nnx.DiffState(0, LoRAParam))(self.model, x, target, self.rngs)
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
        lora_state = nnx.state(self.model, LoRAParam)
        adapter = {
            'rank': self.rank,
            'targets': self.targets,
            'lora_state': lora_state,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(adapter, f)
        print(f"saved LoRA adapter ({self.targets}, rank={self.rank}) to {path}")