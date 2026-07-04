import torch

class MyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Replacing the original pattern (torch.ones_like(x) * 10) 
        # with the similar API: torch.full
        # This tests if requires_grad propagates correctly through torch.full 
        # inside a custom autograd function.
        weight = torch.full(x.shape, 10.0)
        # Save weight for backward pass to compute gradients correctly
        ctx.save_for_backward(weight)
        return torch.matmul(x, weight.t())
    
    @staticmethod
    def backward(ctx, grad_out):
        # Retrieve the saved weight tensor
        weight, = ctx.saved_tensors
        # Compute the gradient for x.
        # Forward: out = x @ weight.t()  (shapes: [2,3] @ [3,2] -> [2,2])
        # Backward: grad_x = grad_out @ weight (shapes: [2,2] @ [2,3] -> [2,3])
        grad_x = torch.matmul(grad_out, weight)
        return grad_x

def test_torch_full_requires_grad_propagation():
    # Input tensor requiring gradient
    x = torch.randn(2, 3, requires_grad=True)
    
    # Run the custom function
    out = MyFn.apply(x)
    
    # Check 1: The output must require grad because the input requires grad
    # and the operation is differentiable with respect to x.
    assert out.requires_grad, "Output should require grad when input requires grad"
    
    # Check 2: Verify backward pass works (graph connectivity)
    out.sum().backward()
    
    # Check 3: Verify gradients were populated
    assert x.grad is not None, "Input x should have gradients after backward pass"
    
    print("Test passed: torch.full preserves requires_grad propagation correctly.")

if __name__ == "__main__":
    test_torch_full_requires_grad_propagation()