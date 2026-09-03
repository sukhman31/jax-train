import os, sys
from flax import nnx

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from inference import LoRAGenerator

BASE_CHECKPOINT_PATH = os.path.join(ROOT, 'checkpoints', 'gpt_shakespeare.pkl')
ADAPTER_PATH = os.path.join(ROOT, 'checkpoints', 'lora_adapter.pkl')

RNGS = nnx.Rngs(0)

generator = LoRAGenerator(BASE_CHECKPOINT_PATH, ADAPTER_PATH, RNGS)
output = generator.generate("To be or", max_new_tokens=100, temperature=0.8)
print(output)
