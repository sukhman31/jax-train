import os, sys
from flax import nnx

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from training import Trainer

CONFIG = {
    'vocab_size':      2256,
    'embed_dim':       384,
    'max_seq_len':     256,
    'num_heads':       6,
    'input_dim':       384,
    'output_dim':      384,
    'scaling_factor':  4,
    'dropout':         0.1,
    'eps':             1e-5,
    'num_of_blocks':   6,
    'max_merges':      2000,
    'learning_rate':   1e-3,
    'seq_len':         256,
    'batch_size':      32,
    'stride':          128,
    'num_epochs':      100,
}

FILE_PATH = os.path.join(ROOT, 'resources', 'tinystories.txt')
CHECKPOINT_PATH = os.path.join(ROOT, 'checkpoints', 'gpt_tinystories.pkl')

RNGS = nnx.Rngs(42)

trainer = Trainer(CONFIG, RNGS)
trainer.train(FILE_PATH, num_epochs=CONFIG['num_epochs'])
trainer.save(CHECKPOINT_PATH)
