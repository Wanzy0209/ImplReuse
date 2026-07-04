import torch
import functools
from enum import Enum

# Handle missing CheckpointPolicy for compatibility with older PyTorch versions
try:
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
except ImportError:
    # Mock CheckpointPolicy if it's not available in the current PyTorch version
    class CheckpointPolicy(Enum):
        MUST_SAVE = 1
        DO_NOT_RECOMPUTE = 2
        PREFER_RECOMPUTE = 3
    
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

class CustomPolicy:
    def __init__(self):
        super().__init__()

    def __call__(self, ctx, out, func, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE

# Define a list of functions to be executed sequentially
def func1(x):
    return x * 2

def func2(x):
    return x + 1

def func3(x):
    return torch.matmul(x, x)

functions = [func1, func2, func3]

# The problematic context_fn using functools.partial
context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())

@torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
def g(x):
    return torch.utils.checkpoint.checkpoint_sequential(
        functions,
        segments=1,
        input=x,
        use_reentrant=False,
        context_fn=context_fn1,
    )

a = torch.randn(4, 4, requires_grad=True, device="cpu")
g(a).sum().backward()