import torch
from torch import nn
import pytest

def test_nested_tensor_narrow_backward_with_anomaly_detection():
    """
    Test case for Issue 161818.
    Verifies that the backward pass works correctly for NestedTensor operations
    (specifically narrow, contiguous, and values) when anomaly detection is enabled.
    """
    # Setup the module and inputs as per the bug report
    module = nn.Linear(8, 12)
    padded = torch.rand(9, 8)
    lengths = torch.as_tensor([5, 4])

    # The bug specifically occurred with anomaly detection enabled
    with torch.autograd.set_detect_anomaly(True):
        # Forward pass
        out = module(padded)
        
        # Perform the nested tensor operations that triggered the bug
        # torch.nested.narrow -> .contiguous() -> .values()
        nopad = torch.nested.narrow(out, dim=1, start=0, length=lengths, layout=torch.jagged).contiguous().values()
        
        # Backward pass
        # This previously raised: NotImplementedError: aten::_is_any_true.default
        try:
            nopad.sum().backward()
        except NotImplementedError as e:
            pytest.fail(f"Backward pass raised NotImplementedError: {e}")

    # Assertions to verify the operation completed successfully
    assert module.weight.grad is not None, "Gradients for weight should not be None"
    assert module.weight.grad.shape == module.weight.shape, "Gradient shape should match weight shape"
    assert module.bias.grad is not None, "Gradients for bias should not be None"

if __name__ == "__main__":
    test_nested_tensor_narrow_backward_with_anomaly_detection()
    print("Test passed successfully.")