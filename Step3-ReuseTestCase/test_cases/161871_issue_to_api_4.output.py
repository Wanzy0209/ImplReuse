import torch
import pytest

def test_mvlgamma_fpe_issue_161871():
    """
    Test case for Issue 161871: Floating point exception in torch.Tensor.mvlgamma_
    
    The bug report indicates a Floating point exception (core dumped) when calling
    mvlgamma_ with a large 'p' value (1024) on an int64 tensor with small random values.
    
    This test verifies that the operation handles the input gracefully (e.g., by raising
    an error or returning NaNs) instead of causing a process crash.
    """
    # Reproduce the tensor creation from the bug report
    tensor = torch.randint(low=0, high=10, size=(5,), dtype=torch.int64)
    
    # Mimic the argument unpacking pattern used in the bug report
    # input = [[tensor, 1024], {}]
    args = [tensor, 1024]
    kwargs = {}
    
    # The bug is a Floating Point Exception (SIGFPE).
    # We expect the API to handle this without crashing the interpreter.
    # mvlgamma_ is an in-place operation.
    
    try:
        # Call the method using the unpacking pattern from the issue
        torch.Tensor.mvlgamma_(*args, **kwargs)
        
        # If the call succeeds, mathematically the result should be NaN or -Inf
        # because the input values (0-9) are less than (p-1)/2 (511.5).
        assert torch.isnan(tensor).all() or torch.isinf(tensor).all()
        
    except (RuntimeError, ValueError, TypeError):
        # If PyTorch correctly identifies the invalid input (int64 or domain error)
        # and raises an exception, that is also an acceptable fix for the crash.
        pass