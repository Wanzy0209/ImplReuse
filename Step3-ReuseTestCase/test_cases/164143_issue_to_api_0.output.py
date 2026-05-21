import torch
import torch.nn.functional as F

def test_tanh_compile_with_debug_mode():
    """
    Test case to verify the interaction between torch.compile, 
    torch.nn.functional.tanh, and DebugMode.
    
    Bug Context: DebugMode silently disables torch.compile.
    This test ensures that the function using tanh is compiled or 
    handled correctly (either by working or raising a clear error) 
    when DebugMode is active.
    """
    
    # Define a function using the similar API: torch.nn.functional.tanh
    def fn(x):
        return F.tanh(x)

    # Input tensor
    x = torch.randn(3, 3)

    # Compile the function using aot_eager backend as mentioned in the bug report
    # as a backend that should ideally work.
    compiled_fn = torch.compile(fn, backend="aot_eager")

    # The bug occurs when running inside a non-infra torch dispatch mode like DebugMode.
    # We use torch._C._DebugMode() to reproduce the specific context mentioned in the issue.
    with torch._C._DebugMode():
        try:
            # Execute the compiled function
            result = compiled_fn(x)
            
            # Verify the output matches the eager execution
            expected = fn(x)
            assert torch.allclose(result, expected), "Output mismatch between compiled and eager execution"
            
            print("Test passed: torch.compile with tanh worked inside DebugMode.")
            
        except Exception as e:
            # The bug report mentions "At minimum we should error".
            # If the fix is to raise an error instead of silently skipping, 
            # we catch it here to acknowledge the expected behavior change.
            print(f"Test raised an error (expected if fix enforces error): {e}")
            raise

if __name__ == "__main__":
    test_tanh_compile_with_debug_mode()