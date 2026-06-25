import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np

class DummyDataset(Dataset):
    def __len__(self):
        return 100
    def __getitem__(self, idx):
        return torch.randn(3, 224, 224)

class InfiniteSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
    def __iter__(self):
        while True:
            yield from torch.randperm(len(self.data_source)).tolist()

dataset = DummyDataset()
sampler = InfiniteSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, num_workers=0, pin_memory=False)

# This will leak memory on Windows
iterator = iter(dataloader)
for i in range(10000):
    batch = next(iterator)
    if i % 100 == 0:
        print(f'Step {i}')