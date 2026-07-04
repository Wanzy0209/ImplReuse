# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch.utils.data import Dataset
from torch import set_default_device
from torch.utils.data import random_split


class TestDataset(Dataset): # basic Dataset for testing
        def __init__(self, x: torch.Tensor, y: torch.Tensor):
            super().__init__()

            self.x = x
            self.y = y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]


def test_bug(device: str = 'cuda'):
    set_default_device(device) # if device is 'cuda' then 'random_split' throws an error

    x = torch.randn(100, 3)
    y = torch.randn(100, 2)

    dataset = TestDataset(x, y)

    train_dataset, val_dataset, test_dataset = random_split( # BUG
        dataset, 
        [0.7, 0.2, 0.1],
        # generator=torch.Generator().manual_seed(42) # also fails with an explicitly set generator
    )

    print(f"Device {device} worked.")

test_bug(device='cpu') # works
test_bug(device='cuda') # throws an error