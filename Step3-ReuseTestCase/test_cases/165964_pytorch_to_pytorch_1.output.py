import torch

# Test case for torch.zeros adapted from the torch.ones bug report
# Verifies if torch.zeros triggers the same CUDA out of memory error
if torch.cuda.is_available():
    # Create a tensor of zeros on CUDA
    tensor = torch.zeros(1, device="cuda")
    # Access the item to force synchronization and memory allocation
    value = tensor.item()
    # Assert the value is correct
    assert value == 0.0
    print("Test passed.")
else:
    print("CUDA not available, skipping test.")