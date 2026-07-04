import torch

# Define a function that uses torch.prod, the similar API identified
def prod_func(x):
    # Performing a product reduction along the last dimension
    return torch.prod(x, dim=-1)

# Check if torch.compile is available (requires PyTorch 2.0+)
if hasattr(torch, 'compile'):
    # Compile with the same backend and settings as the original bug report
    # to check if the compilation failure affects this similar API path
    compiled_prod = torch.compile(prod_func, fullgraph=True, backend="inductor")
else:
    print("torch.compile is not available in this environment. Running in eager mode.")
    compiled_prod = prod_func

with torch.device("cuda"):
    # Use the same tensor shape and dtype as the original 'q' tensor
    # to maintain the context of the bug (B200, bfloat16, large tensors)
    x = torch.randn([2, 32, 4096, 128], dtype=torch.bfloat16, requires_grad=True)

# Run forward pass
y = compiled_prod(x)

# Run backward pass (the original bug occurred during backward compilation)
y.backward(torch.randn_like(y))

# Assertions to verify the test ran successfully
assert torch.isfinite(y).all(), "Forward output contains NaN or Inf"
assert x.grad is not None, "Gradient was not computed"
assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf"

print("Test passed.")