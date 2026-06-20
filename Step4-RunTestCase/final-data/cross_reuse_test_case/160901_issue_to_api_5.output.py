import torch

# Test case adapted from Issue 160901, replacing torch.ones_like with torch.rand_like
# to verify requires_grad propagation behavior in custom autograd functions.

class MyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Replaced torch.ones_like with torch.rand_like to test the similar API
        return torch.matmul(x, (torch.rand_like(x) * 10).t())
    
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out

def test_rand_like_autograd_grad_propagation():
    # Setup input with requires_grad
    x = torch.randn(10, 10, requires_grad=True)
    
    # Test in eager mode
    out = MyFn.apply(x)
    assert out.requires_grad, "Output should require grad in eager mode"
    out.sum().backward()
    assert x.grad is not None, "Gradient should exist in eager mode"
    
    # Test in compiled mode (Dynamo) as the issue mentions "traced"
    # Check if torch.compile is available (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("Skipping compiled mode test: torch.compile is not available (requires PyTorch 2.0+)")
        return

    x_c = x.clone().detach().requires_grad_(True)
    compiled_fn = torch.compile(MyFn.apply)
    
    out_c = compiled_fn(x_c)
    
    # Check for the bug: requires_grad propagation failure
    assert out_c.requires_grad, "Output should require grad in compiled mode"
    
    out_c.sum().backward()
    assert x_c.grad is not None, "Gradient should exist in compiled mode"

if __name__ == "__main__":
    test_rand_like_autograd_grad_propagation()
    print("Test passed.")