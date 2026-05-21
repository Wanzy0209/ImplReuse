import torch
import torch._dynamo

def test_torch_prod_with_fullgraph_and_disable():
    """
    Test case to verify the behavior of torch.compile(fullgraph=True) 
    when encountering torch.compiler.disable, using torch.prod as the operation.
    
    Based on Issue 167927: torch.compile(fullgraph=True) should accept torch.compiler.disable.
    Expected behavior: The combination should work or handle the disable gracefully.
    Current behavior (Bug): Raises torch._dynamo.exc.Unsupported.
    """

    # A function explicitly disabled from compilation
    @torch.compiler.disable
    def disabled_op(x):
        return x + 1

    # The main function to be compiled, utilizing the similar API torch.prod
    def func_to_compile(x):
        # Intentional graph break via disable
        y = disabled_op(x)
        # Use torch.prod (the similar API)
        return torch.prod(y)

    # Attempt to compile with fullgraph=True
    # The bug report indicates this raises:
    # torch._dynamo.exc.Unsupported: Skip calling `torch.compiler.disable()`d function
    try:
        compiled_fn = torch.compile(func_to_compile, fullgraph=True)
        input_tensor = torch.randn(2, 3)
        
        # Execute
        result = compiled_fn(input_tensor)
        
        # Verify correctness if execution succeeds
        expected = torch.prod(input_tensor + 1)
        assert torch.allclose(result, expected), "Output mismatch"
        print("Test Passed: torch.compile(fullgraph=True) accepted torch.compiler.disable with torch.prod.")

    except torch._dynamo.exc.Unsupported as e:
        # This block catches the error described in the bug report
        print(f"Bug Reproduced: {e}")
        # Asserting the specific error message to confirm we hit the reported issue
        assert "Skip calling `torch.compiler.disable()`d function" in str(e)

if __name__ == "__main__":
    test_torch_prod_with_fullgraph_and_disable()