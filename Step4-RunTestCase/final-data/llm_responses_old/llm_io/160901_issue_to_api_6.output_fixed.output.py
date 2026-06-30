import sys
import torch

# Handle import errors for older PyTorch versions where torch.library.impl_abstract does not exist.
try:
    from torch.library import impl_abstract, custom_op
except ImportError:
    print("Skipping test: 'torch.library.impl_abstract' and 'custom_op' are not available. "
          "This test requires PyTorch >= 2.0.")
    sys.exit(0)

# Define a custom operator that mimics the behavior of the autograd.Function in the bug report.
# This replaces the legacy torch.autograd.Function with the modern torch.library API.
@custom_op("test_ns::my_custom_fn", mutates_args=())
def my_custom_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass implementation.
    Mimics: torch.matmul(x, (torch.ones_like(x) * 10).t())
    """
    return torch.matmul(x, (torch.ones_like(x) * 10).t())

# Register the abstract implementation (meta kernel) using the Similar API: torch.library.impl_abstract.
# This is critical for Dynamo tracing. The bug report indicates that requires_grad propagation
# fails during tracing. A correct impl_abstract must ensure that if the input requires grad,
# the output meta tensor also reflects that property.
@impl_abstract("test_ns::my_custom_fn")
def my_custom_fn_abstract(x):
    # Infer the output shape based on the operation logic.
    # Input x: (N, M), Second term: (M, N) -> Output: (N, N)
    # We use new_empty to create a tensor with the correct shape and dtype.
    # Crucially, new_empty preserves the device and other properties like requires_grad
    # if not explicitly overridden, which is the behavior we want to test/verify.
    return x.new_empty((x.size(0), x.size(0)))

# Register the backward pass to ensure the operator is differentiable.
@my_custom_fn.register_backward
def my_custom_fn_backward(grad_out):
    # Mimics the simple backward pass from the bug report: return grad_out
    return grad_out

def test_requires_grad_propagation():
    """
    Test case to verify that requires_grad is correctly propagated through
    the custom operator when using torch.library.impl_abstract, especially
    under torch.compile (Dynamo).
    """
    # Check for torch.compile availability (also a PyTorch 2.0+ feature)
    if not hasattr(torch, 'compile'):
        print("Skipping test: torch.compile is not available (requires PyTorch >= 2.0).")
        return

    # Create an input tensor with requires_grad=True
    x = torch.randn(4, 4, requires_grad=True)

    # 1. Verify behavior in Eager mode
    out_eager = my_custom_fn(x)
    assert out_eager.requires_grad, "Eager mode: Output should require grad"

    # 2. Verify behavior in Compiled mode (Dynamo)
    # The bug report mentions "traced autograd.Function silently incorrect".
    # Dynamo relies on the abstract implementation (impl_abstract) to trace the graph.
    # If impl_abstract does not handle requires_grad correctly, the compiled output
    # will not have requires_grad=True, breaking backpropagation.
    compiled_fn = torch.compile(my_custom_fn)
    out_compiled = compiled_fn(x)
    
    assert out_compiled.requires_grad, (
        "Compiled mode: Output should require grad. "
        "If this fails, the impl_abstract is not propagating requires_grad correctly."
    )

    # 3. Verify the Abstract implementation directly (Meta Tensor)
    # This directly tests the logic registered via torch.library.impl_abstract.
    meta_x = torch.empty(4, 4, device='meta', requires_grad=True)
    meta_out = my_custom_fn(meta_x)
    assert meta_out.requires_grad, "Meta mode: Output should require grad"

    print("Test passed: requires_grad propagated correctly.")

if __name__ == "__main__":
    test_requires_grad_propagation()