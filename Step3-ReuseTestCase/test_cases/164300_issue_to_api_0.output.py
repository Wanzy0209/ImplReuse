import torch
import functools
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

# Define the custom policy used in the checkpoint context
class CustomPolicy:
    def __init__(self):
        super().__init__()

    def __call__(self, ctx, out, func, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE

# Define a simple function to be checkpointed
def f(x, y):
    return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

# Define the context_fn using functools.partial (The reported bug scenario)
context_fn_partial = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())

# Define the context_fn using a standard wrapper (Pattern similar to tf.compat.v1.user_ops.my_fact)
# This serves as a control case to verify the setup works with the expected pattern.
def context_fn_wrapper():
    """Wrapper function mimicking the structure of tf.compat.v1.user_ops.my_fact."""
    return create_selective_checkpoint_contexts(CustomPolicy())

def test_dynamo_sac_partial_context():
    """
    Test case to verify that functools.partial'ed context_fn is supported
    in torch.utils.checkpoint.checkpoint under torch.compile.
    
    This test reproduces the logic from Issue 164300.
    """
    # Setup inputs
    a = torch.randn(4, 4, requires_grad=True, device="cpu")
    b = torch.randn(4, 4, requires_grad=True, device="cpu")

    # Test with functools.partial (Bug reproduction)
    @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
    def g_partial(x, y):
        return torch.utils.checkpoint.checkpoint(
            f, x, y,
            use_reentrant=False,
            context_fn=context_fn_partial,
        )

    try:
        output = g_partial(a, b)
        # Check if output is computed
        assert output is not None
        assert output.shape == (4, 4)
        
        # Check backward pass
        output.sum().backward()
        assert a.grad is not None
        assert b.grad is not None
        print("Test with functools.partial passed.")
    except Exception as e:
        print(f"Test with functools.partial failed: {e}")
        raise

    # Test with standard wrapper (Similar API pattern - tf.compat.v1.user_ops.my_fact)
    # This ensures the "similar API" pattern (def wrapper) continues to work
    @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
    def g_wrapper(x, y):
        return torch.utils.checkpoint.checkpoint(
            f, x, y,
            use_reentrant=False,
            context_fn=context_fn_wrapper,
        )

    try:
        output = g_wrapper(a, b)
        assert output is not None
        assert output.shape == (4, 4)
        output.sum().backward()
        print("Test with wrapper function passed.")
    except Exception as e:
        print(f"Test with wrapper function failed: {e}")
        raise

if __name__ == "__main__":
    test_dynamo_sac_partial_context()