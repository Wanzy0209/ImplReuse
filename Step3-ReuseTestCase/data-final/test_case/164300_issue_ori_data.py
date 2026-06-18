import torch
import functools
from torch.utils.checkpoint import CheckpointPolicy

class CustomPolicy:
    def __init__(self):
        super().__init__()

    def __call__(self, ctx, out, func, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE

from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

def f(x, y):
    return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())

def context_fn2():
    return create_selective_checkpoint_contexts(CustomPolicy())

@torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
def g(x, y):
    return torch.utils.checkpoint.checkpoint(
        f, x, y,
        use_reentrant=False,
        context_fn=context_fn1,
    )

a = torch.randn(4, 4, requires_grad=True, device="cpu")
b = torch.randn(4, 4, requires_grad=True, device="cpu")
g(a, b).sum().backward()