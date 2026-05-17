import torch
from torch.autograd.graph import get_gradient_edge

# Define the custom autograd function from the bug report
class MyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.matmul(x, (torch.ones_like(x) * 10).t())
    
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out

def test_custom_function_gradient_edge():
    # Setup input tensor with requires_grad=True, similar to the bug report
    x = torch.randn(5, 5, requires_grad=True)
    
    # Apply the custom autograd function
    out = MyFn.apply(x)
    
    # Use get_gradient_edge to verify that the autograd graph is correctly connected.
    # The bug report mentions "bad requires_grad propagation". This API call
    # validates that the output tensor 'out' has the correct gradient edge
    # and requires_grad property.
    edge = get_gradient_edge(out)
    
    # Assertions to ensure correctness
    assert out.requires_grad, "Output should require gradients based on input"
    assert edge.node is not None, "Gradient edge node should not be None"
    
    # Verify that get_gradient_edge correctly raises an error if requires_grad is False
    # (simulating a case where propagation might have failed)
    out_detached = out.detach()
    try:
        get_gradient_edge(out_detached)
        raise AssertionError("Expected RuntimeError when getting gradient edge for detached tensor")
    except RuntimeError as e:
        assert "does not require gradients" in str(e)

if __name__ == "__main__":
    test_custom_function_gradient_edge()
    print("Test passed.")