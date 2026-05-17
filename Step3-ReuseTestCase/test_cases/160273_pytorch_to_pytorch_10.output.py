import torch
import torch.nn as nn

def test_lazy_linear_gradient():
    """
    Test case to verify gradient behavior for torch.nn.LazyLinear.
    Adapted from the torch.min gradient test to ensure gradients 
    flow correctly through the lazy-initialized layer.
    """
    # Setup input tensor
    # Using CPU to ensure the test is runnable on all environments
    x = torch.ones([5, 10], requires_grad=True)
    
    # Instantiate LazyLinear
    # in_features is inferred from the input size during the first forward pass
    layer = nn.LazyLinear(out_features=3)
    
    # Verify parameters are uninitialized before forward pass
    assert layer.weight is None, "Weight should be None before first forward pass"
    assert layer.bias is None, "Bias should be None before first forward pass"

    # Forward pass
    # This triggers the lazy initialization of weight and bias
    output = layer(x)
    
    # Backward pass
    # We sum the output to get a scalar for backward()
    output.sum().backward()

    # Verify gradients
    # Unlike torch.min (which distributes gradients or selects indices),
    # LazyLinear should compute standard gradients for the input and parameters.
    assert x.grad is not None, "Input gradient (x.grad) should be computed"
    assert layer.weight.grad is not None, "Weight gradient should be computed"
    assert layer.bias.grad is not None, "Bias gradient should be computed"

    # Check shapes
    assert x.grad.shape == x.shape, "Input gradient shape mismatch"
    assert layer.weight.grad.shape == layer.weight.shape, "Weight gradient shape mismatch"
    assert layer.bias.grad.shape == layer.bias.shape, "Bias gradient shape mismatch"

    print("Test passed: Gradients computed correctly for torch.nn.LazyLinear.")

if __name__ == "__main__":
    test_lazy_linear_gradient()