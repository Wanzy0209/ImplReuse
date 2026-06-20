import torch
import pytest

def test_mvlgamma_large_p_error_handling():
    """
    Test case for Issue 161871: Floating point exception in torch.Tensor.mvlgamma_
    
    This test verifies that torch.Tensor.mvlgamma_ handles invalid inputs (specifically
    a large 'p' value) by raising a proper Python exception, rather than causing a 
    Floating Point Exception (FPE) that crashes the interpreter.
    
    This approach leverages the error-handling pattern associated with the similar API
    (tf.errors.OperatorNotAllowedInGraphError), where invalid operations or states
    result in explicit error raising rather than silent failures or crashes.
    """
    # Reproduce the exact setup from the bug report
    tensor = torch.randint(low=0, high=10, size=(5,), dtype=torch.int64)
    p = 1024
    
    # The original bug resulted in "Floating point exception (core dumped)".
    # The expected behavior after the fix is to raise a RuntimeError or ValueError
    # gracefully, similar to how explicit errors are raised in graph execution contexts
    # for invalid operations.
    with pytest.raises((RuntimeError, ValueError)):
        torch.Tensor.mvlgamma_(tensor, p)

if __name__ == "__main__":
    test_mvlgamma_large_p_error_handling()
    print("Test passed: Exception raised correctly instead of crashing.")