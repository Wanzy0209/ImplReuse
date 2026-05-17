import torch

# Define a custom autograd.Function similar to the original bug report,
# but utilizing the similar API (torch.fmax) internally.
class FmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        # Using torch.fmax which decomposes to torch.where
        return torch.fmax(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        # Return gradients for both inputs
        return grad_output, grad_output

def test_fmax_traced_grad_propagation():
    # Setup inputs with requires_grad=True
    x = torch.randn(10, 10, requires_grad=True)
    y = torch.randn(10, 10, requires_grad=True)

    # Define the function to be traced
    def func(x, y):
        return FmaxFunction.apply(x, y)

    # Compile the function using torch.compile (dynamo)
    # This mirrors the "traced autograd.Function" context of the bug
    compiled_func = torch.compile(func)

    # Forward pass
    out = compiled_func(x, y)

    # Backward pass
    out.sum().backward()

    # Assertions to check for "bad requires_grad propagation"
    # If the bug exists, x.grad or y.grad might be None or incorrect
    assert x.grad is not None, "Gradient for x is None (requires_grad not propagated)"
    assert y.grad is not None, "Gradient for y is None (requires_grad not propagated)"
    
    # Check that gradients are not all zeros (sanity check for correctness)
    assert torch.any(x.grad != 0), "Gradient for x is all zeros"
    assert torch.any(y.grad != 0), "Gradient for y is all zeros"

if __name__ == "__main__":
    test_fmax_traced_grad_propagation()
    print("Test passed.")