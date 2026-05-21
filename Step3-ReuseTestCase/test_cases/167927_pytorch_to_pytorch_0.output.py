import torch
import torch._dynamo

def test_compile_fullgraph_with_compiler_disable():
    """
    Test that torch.compile(fullgraph=True) accepts torch.compiler.disable.
    
    Previously, using torch.compiler.disable inside a function compiled with
    fullgraph=True would raise torch._dynamo.exc.Unsupported.
    This test verifies that the combination works as intended.
    """
    
    def fn(x):
        x = x + 1
        # Intentional graph break using torch.compiler.disable
        with torch.compiler.disable():
            # This block runs in eager mode
            x = x * 2
        x = x + 3
        return x

    # Compile with fullgraph=True
    # Before the fix, this raises: torch._dynamo.exc.Unsupported: Skip calling `torch.compiler.disable()`d function
    compiled_fn = torch.compile(fn, fullgraph=True)

    input_tensor = torch.randn(3, 3)
    
    # Run the compiled function
    result = compiled_fn(input_tensor)
    
    # Run the original function for comparison
    expected = fn(input_tensor)
    
    # Verify the results match
    assert torch.allclose(result, expected), "Output mismatch between compiled and eager execution"
    print("Test passed: torch.compile(fullgraph=True) accepts torch.compiler.disable")

if __name__ == "__main__":
    test_compile_fullgraph_with_compiler_disable()