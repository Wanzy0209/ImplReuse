import torch

# Test torch.manual_seed functionality
# Set a specific seed to ensure reproducibility
torch.manual_seed(42)

# Generate a random tensor
tensor1 = torch.rand(5)

# Reset the seed to the same value
torch.manual_seed(42)

# Generate another random tensor
tensor2 = torch.rand(5)

# Verify that the generated tensors are identical
assert torch.equal(tensor1, tensor2), "torch.manual_seed did not produce reproducible results"