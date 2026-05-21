import torch
import torch._dynamo
from torch.testing._internal.debug_mode import DebugMode

def test_torch_prod_compile_with_debug_mode():
    """
    Test case to verify that torch.compile does not silently disable compilation
    when DebugMode is active, using torch.prod as the target operation.
    """
    # Define a function using the similar API (torch.prod)
    def fn(x):
        return torch.prod(x)

    # Reset dynamo counters to ensure a clean test environment
    torch._dynamo.reset()

    # Enable DebugMode, which is the context triggering the bug
    with DebugMode():
        # Attempt to compile the function
        # The bug report suggests using aot_eager as a backend that should ideally work
        compiled_fn = torch.compile(fn, backend="aot_eager")
        
        x = torch.randn(3, 3)
        
        # Execute the compiled function
        result = compiled_fn(x)
        
        # Verify the output is correct
        expected = fn(x)
        assert torch.allclose(result, expected), "Output mismatch between compiled and eager execution"
        
        # Verify that compilation actually occurred.
        # If the bug (silent disable) is present, this assertion will fail 
        # because the frame was skipped.
    torch._dynamo.assert_compiled(fn)

if __name__ == "__main__":
    test_torch_prod_compile_with_debug_mode()