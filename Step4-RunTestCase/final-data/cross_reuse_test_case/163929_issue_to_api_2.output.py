import torch
import pytest

# Check if torch.compile is available (requires PyTorch >= 2.0)
# If not available, skip the test to avoid AttributeError
if not hasattr(torch, 'compile'):
    pytest.skip("torch.compile is not available (requires PyTorch >= 2.0)", allow_module_level=True)

def test_argmin_argmax_transposed_mutated():
    """
    Test case for Issue 163929.
    
    Verifies that torch.compile (inductor backend) produces correct results
    for argmin and argmax operations on tensors that have been mutated
    in-place and then transposed.
    """
    torch.manual_seed(0)

    # Test case for argmin (from the code snippet in the issue)
    def func_argmin(x):
        x.tan_()
        x = x.t()
        return x.argmin()

    x1 = torch.randn(4, 6)
    x2 = x1.clone()
    
    # Eager execution
    out_eager = func_argmin(x1)
    
    # Compiled execution (inductor)
    compiled_func = torch.compile(func_argmin)
    out_compiled = compiled_func(x2)
    
    torch.testing.assert_close(out_eager, out_compiled)

    # Test case for argmax (from the title of the issue)
    def func_argmax(x):
        x.tan_()
        x = x.t()
        return x.argmax()

    x3 = torch.randn(4, 6)
    x4 = x3.clone()
    
    out_eager_max = func_argmax(x3)
    compiled_func_max = torch.compile(func_argmax)
    out_compiled_max = compiled_func_max(x4)
    
    torch.testing.assert_close(out_eager_max, out_compiled_max)

if __name__ == "__main__":
    test_argmin_argmax_transposed_mutated()
    print("Test passed.")