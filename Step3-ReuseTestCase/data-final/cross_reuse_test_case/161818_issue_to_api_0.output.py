import torch
from torch import nn

def test_nested_tensor_backward_with_sigmoid():
    """
    Test case adapted from Issue 161818.
    Replaces nn.Linear with the similar API torch.nn.Sigmoid to verify
    backward pass stability with NestedTensor operations.
    """
    # Use the similar API: nn.Sigmoid
    module = nn.Sigmoid()
    
    # Setup input data matching the original issue dimensions
    padded = torch.rand(9, 8, requires_grad=True)
    lengths = torch.as_tensor([5, 4])

    # Enable anomaly detection to catch errors during backward pass
    # (as done in the original bug report)
    with torch.autograd.set_detect_anomaly(True):
        # Forward pass through the module
        out = module(padded)
        
        # Apply the NestedTensor narrow operation (the problematic part from the issue)
        # Preserving dim=1 and lengths=[5, 4] from the original logic
        nopad = torch.nested.narrow(
            out, dim=1, start=0, length=lengths, layout=torch.jagged
        ).contiguous().values()
        
        # Perform backward pass
        nopad.sum().backward()

    # Verify gradients are computed and shapes match
    assert padded.grad is not None, "Gradient for input tensor 'padded' is None"
    assert padded.grad.shape == padded.shape, f"Gradient shape mismatch: {padded.grad.shape} vs {padded.shape}"
    
    print("Test passed: Backward pass completed successfully with Sigmoid.")

if __name__ == "__main__":
    test_nested_tensor_backward_with_sigmoid()