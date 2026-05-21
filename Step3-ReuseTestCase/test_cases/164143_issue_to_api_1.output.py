import torch
import torch.nn.functional as F
from torch.testing._internal.debug_mode import DebugMode

def test_torch_compile_with_debug_mode():
    """
    Test case for Issue 164143: DebugMode silently disables torch.compile.
    
    This test verifies that torch.compile raises an error when used inside
    DebugMode (a non-infra torch dispatch mode), rather than silently 
    skipping compilation. The function being compiled uses 
    torch.nn.functional.sigmoid as the operation.
    """
    # Define a function using the similar API: torch.nn.functional.sigmoid
    def model(x):
        return F.sigmoid(x)

    x = torch.randn(2, 2)

    # The bug report indicates that DebugMode causes torch.compile to skip silently.
    # The requested fix is to raise an error when this occurs.
    with DebugMode():
        compiled_model = torch.compile(model, backend="aot_eager")
        
        try:
            compiled_model(x)
            # If we reach here, the bug might still exist (silent skip) or it might be working.
            # However, the issue explicitly requests: "At minimum we should error".
            # Therefore, we assert that an error should have been raised.
            raise AssertionError("Expected RuntimeError when using torch.compile with DebugMode, but it silently skipped or succeeded.")
        except RuntimeError as e:
            # Check if the error message is related to the dispatch mode conflict
            assert "non-infra torch dispatch mode" in str(e) or "DebugMode" in str(e)

if __name__ == "__main__":
    test_torch_compile_with_debug_mode()