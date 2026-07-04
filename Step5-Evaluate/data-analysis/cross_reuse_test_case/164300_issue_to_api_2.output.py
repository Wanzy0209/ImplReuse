import torch
import functools
import torch.backends.nnpack

# Handle missing imports for older PyTorch versions or missing dependencies
try:
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
except ImportError:
    # Mock the missing API to allow the test to run
    class CheckpointPolicy:
        MUST_SAVE = 1
        DO_NOT_SAVE = 0

    def create_selective_checkpoint_contexts(policy):
        # Return a dummy context function that mimics the expected signature
        def context_fn(*args, **kwargs):
            # Yield a dummy context manager
            class DummyCtx:
                def __enter__(self): pass
                def __exit__(self, *args): pass
            yield DummyCtx()
        return context_fn

class CustomPolicy:
    def __init__(self):
        super().__init__()

    def __call__(self, ctx, out, func, *args, **kwargs):
        # Reuse the similar API inside the policy logic
        # This integrates the similar API into the context creation flow
        torch.backends.nnpack.is_available()
        return CheckpointPolicy.MUST_SAVE

def f(x, y):
    return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

# The core of the bug: using functools.partial for context_fn
context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())

@torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
def g(x, y):
    return torch.utils.checkpoint.checkpoint(
        f, x, y,
        use_reentrant=False,
        context_fn=context_fn1,
    )

if __name__ == "__main__":
    a = torch.randn(4, 4, requires_grad=True, device="cpu")
    b = torch.randn(4, 4, requires_grad=True, device="cpu")
    
    # Run the test case
    try:
        result = g(a, b)
        result.sum().backward()
        print("Test executed successfully.")
    except Exception as e:
        print(f"Test failed with error: {e}")