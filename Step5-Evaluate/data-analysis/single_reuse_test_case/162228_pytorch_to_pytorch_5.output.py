import torch

# Fix for environments with PyTorch < 2.0 where torch.compile is not available
if not hasattr(torch, 'compile'):
    print("Warning: torch.compile is not available (requires PyTorch >= 2.0). Using a pass-through decorator.")
    def dummy_compile(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    torch.compile = dummy_compile

# Adapted test case for torch.any based on the flex_attention backpropagation issue
# The original issue involves graph breaks and gradient flow through a closure.
# Here we test if torch.any (a similar API in terms of being a reduction/op) 
# handles gradients and graph breaks correctly inside torch.compile.

@torch.compile()
def test(x, y):
    # Use torch.any to create a mask from y
    # This replaces the flex_attention call and the score_mod logic
    # We check if the dependency on y is handled correctly
    mask = torch.any(y > 0, dim=1, keepdim=True).float() # Shape (B, 1, 1)

    # Apply mask to x
    # torch._dynamo.graph_break()
    return x * mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, L, D = 2, 16, 64

x = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
y = torch.randn(B, L, device=DEVICE, requires_grad=True)

out = test(x, y).mean().backward()

print(torch.__version__)
# x should have gradients as it is scaled by the mask
# y is used in torch.any, which is non-differentiable, so y.grad should be None or 0
print(f"x: {(x.grad is not None) and (x.grad.norm() > 0)}, y: {(y.grad is not None) and (y.grad.norm() > 0)}")

assert x.grad is not None, "x.grad is None"
assert x.grad.norm() > 0, "x.grad is zero"

# For torch.any, we don't expect gradients to flow to y, unlike the original flex_attention case.
# We verify that the execution completes without error.
if y.grad is not None:
    assert y.grad.norm() == 0, "y.grad should be zero for torch.any"
else:
    print("y.grad is None as expected for non-differentiable op")