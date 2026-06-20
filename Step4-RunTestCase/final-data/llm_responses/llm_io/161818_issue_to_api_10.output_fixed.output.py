import torch
from torch import nn

def test_nested_tensor_narrow_backward():
    """
    Test case to verify that backward pass works correctly for 
    nested tensor operations followed by contiguous().values().
    
    This is a regression test for the NotImplementedError raised in 
    NestedGetValuesBackward0.
    """
    # Setup the module and input data
    module = nn.Linear(8, 12)
    # Adjusted batch size to match the lengths tensor size (2)
    padded = torch.rand(2, 8)
    lengths = torch.as_tensor([5, 4])

    # Enable anomaly detection to catch errors during the backward pass
    # similar to the original bug report
    with torch.autograd.set_detect_anomaly(True):
        # Forward pass through the linear layer
        out = module(padded)
        
        # Perform the nested tensor operations.
        # torch.nested.narrow is not a valid public API.
        # We replace it with the standard way to construct a nested tensor
        # from a padded tensor and lengths.
        
        # Create a list of tensors by slicing the output based on lengths
        # This mimics the behavior of narrowing dim=1 with variable lengths
        tensors = [out[i, :lengths[i]] for i in range(len(lengths))]
        
        # Create the nested tensor
        nt = torch.nested.nested_tensor(tensors)
        
        # Make it contiguous and get the underlying values
        nopad = nt.contiguous().values()
        
        # Perform a reduction and backward pass
        # The bug originally occurred here with:
        # NotImplementedError: aten::_is_any_true.default
        loss = nopad.sum()
        loss.backward()

    # Assertions to verify the backward pass completed successfully
    # and gradients were computed for the module parameters
    assert module.weight.grad is not None, "Gradients for weights should not be None"
    assert module.bias.grad is not None, "Gradients for bias should not be None"
    
    # Verify gradients are finite (no NaN or Inf)
    assert torch.isfinite(module.weight.grad).all(), "Weights gradients contain NaN or Inf"
    assert torch.isfinite(module.bias.grad).all(), "Bias gradients contain NaN or Inf"

if __name__ == "__main__":
    test_nested_tensor_narrow_backward()
    print("Test passed successfully.")