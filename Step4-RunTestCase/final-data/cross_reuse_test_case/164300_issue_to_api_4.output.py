import torch
import functools

# Handle missing imports for older PyTorch versions or environments where selective checkpointing is not available
try:
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
except ImportError:
    # Mock CheckpointPolicy if it doesn't exist
    class CheckpointPolicy:
        MUST_SAVE = 1
        # Add other policies if necessary, but MUST_SAVE is the only one used in the test

    # Mock create_selective_checkpoint_contexts if it doesn't exist
    def create_selective_checkpoint_contexts(policy):
        # Return a dummy context function that mimics the expected signature
        def dummy_context_fn(ctx):
            return ctx
        return dummy_context_fn

# Leverage the similar API: torch.backends.mkldnn.is_available
# We integrate this into the CustomPolicy to reflect the code similarity
# (simple return logic based on a check) and reuse the API.
class CustomPolicy:
    def __call__(self, ctx, out, func, *args, **kwargs):
        # Reuse the similar API to determine the policy logic
        if torch.backends.mkldnn.is_available():
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.MUST_SAVE

def f(x, y):
    return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

# Original bug reproduction logic: functools.partial'ed context_fn
# This is the core of the issue being tested.
context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())

@torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
def g(x, y):
    return torch.utils.checkpoint.checkpoint(
        f, x, y,
        use_reentrant=False,
        context_fn=context_fn1,
    )

def test_checkpoint_with_partial_context_fn():
    """
    Test that torch.utils.checkpoint.checkpoint supports functools.partial'ed context_fn
    when used with torch.compile, while leveraging torch.backends.mkldnn.is_available
    in the policy logic.
    """
    a = torch.randn(4, 4, requires_grad=True, device="cpu")
    b = torch.randn(4, 4, requires_grad=True, device="cpu")
    
    # Run the compiled function
    output = g(a, b)
    
    # Check forward pass
    assert output is not None
    assert output.shape == (4, 4)
    
    # Run backward pass
    output.sum().backward()
    
    # Check gradients
    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == (4, 4)
    assert b.grad.shape == (4, 4)

if __name__ == "__main__":
    test_checkpoint_with_partial_context_fn()
    print("Test passed successfully.")