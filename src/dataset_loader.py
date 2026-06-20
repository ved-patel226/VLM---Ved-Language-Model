import numpy as np
import torch

class BinDataset:
    def __init__(self, path, block_size):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def get_batch(self, batch_size):
        ix = np.random.randint(
            0, len(self.data) - self.block_size -1, 
            size=(batch_size,)
        )

        x = np.stack([self.data[i:i+self.block_size] for i in ix])
        y = np.stack([self.data[i+1:i+self.block_size+1] for i in ix])

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data) // self.block_size