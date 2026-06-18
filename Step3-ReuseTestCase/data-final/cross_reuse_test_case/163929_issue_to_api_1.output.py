import torch
import pytest

# The bug report indicates an issue with inductor when performing in-place mutations,
# transpositions, and reductions (argmin/argmax). The title mentions argmax, 
# the code uses argmin. We should test both to ensure the fix covers the class of issues.
# The similar API provided (torch.backends.cusparselt.version) is unrelated to the functional bug,
# but checking backend availability is a good practice for tests involving compilation.

def test_argmin_argmax_transposed_mutated():
    """
    Test that argmin and argmax produce correct results on transposed, mutated tensors
    when compiled with torch.compile (inductor backend).
    
    This test reproduces the logic from Issue 163929.
    """
    # Check if CUDA is available if we were to strictly follow cusparselt context,
    # but the bug is inductor (CPU/CUDA). We'll run generally.
    # torch.backends.cusparselt.version() # Not strictly needed for the logic, but noted.

    torch.manual_seed(0)
    
    # Test argmin (from the original code)
    def foo_argmin(x):
        x.tan_()
        x = x.t()
        return x.argmin()

    x1 = torch.randn(4, 6)
    x2 = x1.clone()
    
    out1_eager = foo_argmin(x1)
    
    # Compile the function
    cf_argmin = torch.compile(foo_argmin)
    out2_compiled = cf_argmin(x2)
    
    torch.testing.assert_close(out1_eager, out2_compiled)

    # Test argmax (from the title)
    def foo_argmax(x):
        x.tan_()
        x = x.t()
        return x.argmax()

    x3 = torch.randn(4, 6)
    x4 = x3.clone()
    
    out3_eager = foo_argmax(x3)
    
    cf_argmax = torch.compile(foo_argmax)
    out4_compiled = cf_argmax(x4)
    
    torch.testing.assert_close(out3_eager, out4_compiled)

if __name__ == "__main__":
    test_argmin_argmax_transposed_mutated()
    print("Test passed.")