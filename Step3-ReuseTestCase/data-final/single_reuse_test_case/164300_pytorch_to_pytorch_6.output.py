import torch
import functools
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
from torch.export import export

class CustomPolicy:
    def __init__(self):
        super().__init__()

    def __call__(self, ctx, out, func, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE

def f(x, y):
    return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

# Create the context_fn using functools.partial, which was the issue in the original bug
context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())

def g(x, y):
    return torch.utils.checkpoint.checkpoint(
        f, x, y,
        use_reentrant=False,
        context_fn=context_fn1,
    )

# Setup inputs
a = torch.randn(4, 4, requires_grad=True, device="cpu")
b = torch.randn(4, 4, requires_grad=True, device="cpu")

# Test torch.export.export with the partial'ed context_fn
# This verifies if export handles functools.partial arguments correctly
try:
    exported_program = export(g, args=(a, b))
    
    # Verify the exported program runs and computes gradients
    res = exported_program.module()(a, b)
    res.sum().backward()
    
    print("Test passed: torch.export.export supports functools.partial context_fn")
except Exception as e:
    print(f"Test failed: {e}")
    raise