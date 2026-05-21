import torch
from torch import set_default_device

def test_broadcast_shapes(device: str = 'cuda'):
    """
    Test if torch.broadcast_shapes works correctly when the default device is set.
    This is adapted from the bug report for torch.utils.data.random_split.
    """
    set_default_device(device)

    # Define some shapes to broadcast
    shape1 = (3, 1)
    shape2 = (1, 5)
    shape3 = (10, 32, 32)
    shape4 = (10, 1, 32)

    # Call the similar API
    result1 = torch.broadcast_shapes(shape1, shape2)
    result2 = torch.broadcast_shapes(shape3, shape4)

    # Verify results
    assert result1 == torch.Size([3, 5]), f"Expected (3, 5), got {result1}"
    assert result2 == torch.Size([10, 32, 32]), f"Expected (10, 32, 32), got {result2}"

    print(f"Device {device} worked for torch.broadcast_shapes.")

# Test with CPU (baseline)
test_broadcast_shapes(device='cpu')

# Test with CUDA (checking for the reported bug type)
test_broadcast_shapes(device='cuda')