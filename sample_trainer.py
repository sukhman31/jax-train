import os, sys
from flax import nnx

sys.path.insert(0, os.path.dirname(__file__))

from transform import Trainer

CONFIG = {
    'vocab_size':      1000,
    'embed_dim':       64,
    'max_seq_len':     256,
    'num_heads':       4,
    'input_dim':       64,
    'output_dim':      64,
    'scaling_factor':  4,
    'dropout':         0.1,
    'eps':             1e-5,
    'num_of_blocks':   2,
    'max_merges':      744,
    'learning_rate':   1e-3,
    'seq_len':         32,
    'batch_size':      8,
    'num_epochs':      50,
}

FILE_PATH = os.path.join(os.path.dirname(__file__), 'resources', 'shakespeare_sample.txt')
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', 'gpt_shakespeare.pkl')

RNGS = nnx.Rngs(0)

trainer = Trainer(CONFIG, RNGS)
trainer.train(FILE_PATH, num_epochs=CONFIG['num_epochs'])
trainer.save(CHECKPOINT_PATH)
