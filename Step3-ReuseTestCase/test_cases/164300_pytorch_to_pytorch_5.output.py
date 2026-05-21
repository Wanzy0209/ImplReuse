import torch
import functools
import torch.nn.functional as F

# Adapt the test case for torch.nn.functional.upsample
# The original bug involves functools.partial not being supported for a callback argument.
# Since upsample does not take a callback, we test if upsample itself can be wrapped in 
# functools.partial and executed correctly within a torch.compile context.

# Define a partial of the similar API
upsample_partial = functools.partial(F.upsample, scale_factor=2, mode='nearest')

@torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
def g(x):
    # Call the partial function inside the compiled graph
    return upsample_partial(x)

# Create input tensor (Batch, Channel, Height, Width)
a = torch.randn(1, 3, 4, 4, requires_grad=True, device="cpu")

# Run forward pass
output = g(a)

# Run backward pass
output.sum().backward()

# Assertions to verify correctness
assert output.shape == (1, 3, 8, 8), f"Expected shape (1, 3, 8, 8), got {output.shape}"
assert a.grad is not None, "Gradient is None"
assert a.grad.shape == a.shape, f"Gradient shape mismatch: {a.grad.shape} vs {a.shape}"