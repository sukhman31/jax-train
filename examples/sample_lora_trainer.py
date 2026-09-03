import os, sys
from flax import nnx

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from training import LoRATrainer

BASE_CHECKPOINT_PATH = os.path.join(ROOT, 'checkpoints', 'gpt_shakespeare.pkl')
ADAPTER_PATH = os.path.join(ROOT, 'checkpoints', 'lora_adapter.pkl')
FILE_PATH = os.path.join(ROOT, 'resources', 'shakespeare_sample.txt')

RANK = 8
TARGETS = ('q', 'v')
NUM_EPOCHS = 50

RNGS = nnx.Rngs(0)

lora_trainer = LoRATrainer(BASE_CHECKPOINT_PATH, RNGS, rank=RANK, targets=TARGETS)
lora_trainer.train(FILE_PATH, num_epochs=NUM_EPOCHS)
lora_trainer.save(ADAPTER_PATH)
