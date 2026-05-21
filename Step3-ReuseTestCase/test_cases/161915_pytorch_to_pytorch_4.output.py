import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import swa_utils

# Adapted test case for torch.optim.swa_utils.update_bn
# The original issue involved a segmentation fault when calling share_memory_() on a NestedTensor.
# Here we verify that the similar API, update_bn, handles tensor operations correctly without crashing.

# Setup: Create a model with BatchNorm layers (required for update_bn)
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.BatchNorm1d(5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# Setup: Create a data loader with random tensors
# This mimics the random tensor creation in the original bug report
data = torch.randn(100, 10)
targets = torch.randint(0, 2, (100,))
dataset = TensorDataset(data, targets)
loader = DataLoader(dataset, batch_size=10)

# Action: Call the similar API
# This replaces the original call site 'nt.share_memory_()'
swa_utils.update_bn(loader, model)

# Verification: Ensure the operation completed successfully
# (The original bug was a segfault, so reaching this point implies success)
print("Test passed: update_bn executed successfully.")