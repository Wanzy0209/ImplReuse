import torch
import torch.nn.functional as F

def test_sigmoid_zero_dim(device):
    """
    Test case adapted from the torch.var zero-dimensional tensor issue.
    Since torch.nn.functional.sigmoid is an element-wise operation and does not
    support the 'dim' argument, we test the core aspect of the bug report:
    handling zero-dimensional input tensors on different devices.
    """
    # Create a zero-dimensional tensor (scalar) as in the original bug report
    x = torch.tensor(3.0, device=device)
    
    try:
        # Apply sigmoid. We omit 'dim=0' here because sigmoid is element-wise
        # and does not perform reduction along a dimension.
        output = F.sigmoid(x)
        print(f"sigmoid test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"sigmoid test fails for device: {device}: {e}")

# Run tests
test_sigmoid_zero_dim(device="cpu")

if torch.backends.mps.is_available():
    test_sigmoid_zero_dim(device="mps")
else:
    print("MPS device not available, skipping MPS test.")