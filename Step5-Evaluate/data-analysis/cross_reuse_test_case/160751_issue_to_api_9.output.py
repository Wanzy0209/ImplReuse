import torch
import pytest

def test_aot_eager_synchronize_with_logical_or():
    """
    Test that torch.cuda.synchronize() is not removed in aot_eager mode
    when used in conjunction with torch.logical_or.
    
    This test adapts the original bug reproduction logic by replacing
    the simple comparison (a > 0) with a logic chain involving torch.logical_or.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    def func():
        # Create tensors such that the logical_or result will trigger the assertion failure
        a = torch.tensor([1.0, -2.0], device="cuda")
        b = torch.tensor([-1.0, -2.0], device="cuda")
        
        # Use torch.logical_or to combine boolean conditions
        # a > 0 -> [True, False]
        # b > 0 -> [False, False]
        # logical_or -> [True, False]
        # all -> False
        cond1 = a > 0
        cond2 = b > 0
        combined = torch.logical_or(cond1, cond2)
        result = torch.all(combined)
        
        # This assertion should fail
        assert result, "Assertion should be raised"
        
        # If synchronize is removed, the exception handling might be affected
        torch.cuda.synchronize()
        print("This should not run if assertion is caught")

    # Fix: Check if _dynamo attribute exists before calling reset
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
        
    # Compile with aot_eager backend, which is the subject of the bug
    f_c = torch.compile(func, backend="aot_eager")
    
    # We expect an AssertionError because the condition is False.
    # If the bug exists (synchronize removed causing missed exceptions),
    # this test might fail to catch the exception or behave unexpectedly.
    with pytest.raises(AssertionError, match="Assertion should be raised"):
        f_c()

if __name__ == "__main__":
    test_aot_eager_synchronize_with_logical_or()