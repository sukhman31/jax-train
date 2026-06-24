import numpy as np
import jax.numpy as jnp

class SlidingWindowSampler:
    def __init__(self,seq_len,batch_size,token_ids,stride = 1):
        self.seq_len = seq_len
        self.batch_size = batch_size
        inputs, targets = [], []

        for i in range(0,len(token_ids)-seq_len,stride):
            inputs.append(token_ids[i:i+seq_len])
            targets.append(token_ids[i+1:i+seq_len+1])
        
        self.inputs = np.array(inputs)
        self.targets = np.array(targets)
        self.num_batches = len(self.inputs) // batch_size
    
    def shuffle(self):
        idx = np.random.permutation(len(self.inputs))
        self.inputs  = self.inputs[idx]
        self.targets = self.targets[idx]
    
    def __iter__(self):
        for i in range(self.num_batches):
            start = i * self.batch_size
            end   = start + self.batch_size
            yield (
                jnp.array(self.inputs[start:end]),
                jnp.array(self.targets[start:end])
            )