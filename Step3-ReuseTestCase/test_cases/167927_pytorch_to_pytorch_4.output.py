import torch

def test_torch_all_with_fullgraph_disable():
    """
    Test case to verify that torch.all works inside torch.compiler.disable
    when the outer function is compiled with fullgraph=True.
    
    This addresses the issue where torch.compile(fullgraph=True) raises
    torch._dynamo.exc.Unsupported when encountering torch.compiler.disable.
    """
    
    def func(x):
        # Intentionally disable compilation for this block
        with torch.compiler.disable():
            # Use torch.all (the similar API) inside the disabled block
            # to verify its behavior in this specific context.
            check = torch.all(x > 0)
        
        return x + check

    # Compile with fullgraph=True
    # Bug 167927: This currently raises torch._dynamo.exc.Unsupported
    compiled_func = torch.compile(func, fullgraph=True)

    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    
    # Execute the compiled function
    # If the bug is fixed, this should run without raising Unsupported.
    try:
        result = compiled_func(input_tensor)
        expected = func(input_tensor)
        assert torch.equal(result, expected), "Output mismatch"
        print("Test passed: torch.all works with fullgraph=True and disable.")
    except torch._dynamo.exc.Unsupported as e:
        print(f"Test failed (Bug 167927): {e}")
        raise

if __name__ == "__main__":
    test_torch_all_with_fullgraph_disable()