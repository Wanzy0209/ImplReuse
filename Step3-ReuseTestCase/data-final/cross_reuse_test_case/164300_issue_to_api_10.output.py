import torch
import functools
from torch.library import Library, fallthrough_kernel

def test_fallthrough_kernel_with_partial():
    """
    Test that torch.library.impl correctly handles a functools.partial
    wrapping of fallthrough_kernel, similar to the issue where
    torch.utils.checkpoint.checkpoint failed with partial'd context_fn.
    """
    # Define a custom library and operation
    lib = Library("test_fallthrough_lib", "DEF")
    lib.define("test_op(Tensor x) -> Tensor")

    # Reproduce the logic: using functools.partial on the similar API
    # In the original bug, context_fn was partial'd.
    # Here, we partial the fallthrough_kernel.
    partial_fallthrough = functools.partial(fallthrough_kernel)

    # Register the partial'd kernel
    # This tests if the registration mechanism handles partials correctly
    lib.impl("test_op", partial_fallthrough)

    # Define a function to be compiled (similar to the original issue's use of torch.compile)
    @torch.compile(fullgraph=True)
    def run_op(x):
        return lib.test_op(x)

    # Test execution
    x = torch.randn(2, 2)
    try:
        run_op(x)
        # If we reach here, the test might be invalid as fallthrough_kernel should raise
        assert False, "Expected NotImplementedError from fallthrough_kernel"
    except NotImplementedError as e:
        # Expected behavior: fallthrough_kernel raises NotImplementedError
        assert "fallthrough_kernel() should never be called" in str(e)
        print("Test passed: Partial fallthrough_kernel registered and raised expected error.")

if __name__ == "__main__":
    test_fallthrough_kernel_with_partial()