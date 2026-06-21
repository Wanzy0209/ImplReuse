import torch

# Setup from the original bug report
# torch.jagged is not available in the current environment, but nested_tensor
# creates a jagged layout by default when given ragged inputs.
x = torch.nested.nested_tensor([torch.ones(3, 2, 3), torch.ones(4, 2, 3)])

# Test the similar API: torch.chunk
# We test along dimension 0, which caused the crash in torch.cat
try:
    # Attempt to split the jagged tensor into 2 chunks along the jagged dimension
    chunks = torch.chunk(x, chunks=2, dim=0)
    print("torch.chunk succeeded on dim 0")
except Exception as e:
    print(f"torch.chunk failed on dim 0: {e}")

# Test along a non-jagged dimension for comparison
try:
    chunks = torch.chunk(x, chunks=2, dim=1)
    print("torch.chunk succeeded on dim 1")
except Exception as e:
    print(f"torch.chunk failed on dim 1: {e}")