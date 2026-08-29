import os, sys
from flax import nnx

sys.path.insert(0, os.path.dirname(__file__))

from inference import Generator

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', 'gpt_shakespeare.pkl')

RNGS = nnx.Rngs(0)

generator = Generator(CHECKPOINT_PATH, RNGS)
output = generator.generate("To be or", max_new_tokens=100, temperature=0.8)
print(output)
