import torch
import functools
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
import torch.profiler.itt as itt

class CustomPolicy:
    def __init__(self):
        super().__init__()

    def __call__(self, ctx, out, func, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE

def f(x, y):
    # Leveraging the similar API (torch.profiler.itt.range_pop) inside the function
    # to test interaction between checkpointing, partials, and profiling ranges.
    itt.range_push("f_compute")
    res = torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y
    itt.range_pop()
    return res

# The bug reproduction logic: using functools.partial for context_fn
context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())

@torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
def g(x, y):
    return torch.utils.checkpoint.checkpoint(
        f, x, y,
        use_reentrant=False,
        context_fn=context_fn1,
    )

def test_checkpoint_with_partial_and_profiling():
    a = torch.randn(4, 4, requires_grad=True, device="cpu")
    b = torch.randn(4, 4, requires_grad=True, device="cpu")
    
    # Run the forward and backward pass
    # This should not raise an error if the bug is fixed
    output = g(a, b)
    loss = output.sum()
    loss.backward()

    # Assertions to verify execution
    assert output is not None
    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape

if __name__ == "__main__":
    test_checkpoint_with_partial_and_profiling()
    print("Test passed.")