import torch

def test_torch_all_compile_regression():
    """
    Test case for torch.all inside torch.compile.
    Based on Issue 161372 regarding recompilation limits and size mismatches.
    This test verifies that torch.all works correctly within a compiled function
    when input shapes vary dynamically, which previously triggered recompilation limits.
    """

    def func_with_all(x):
        # torch.all is used here to determine control flow.
        # In the original bug, dynamic shapes caused cache size mismatches.
        # We check if all elements are positive to decide the operation.
        if torch.all(x > 0):
            return x * 2.0
        else:
            return x * -1.0

    # Compile the function
    compiled_func = torch.compile(func_with_all)

    # The original bug report mentioned a size mismatch (expected 77, actual 78).
    # We simulate this scenario by iterating through a range of sizes around that value.
    # This tests if the compiler handles dynamic shapes correctly without hitting
    # the recompile limit or throwing size mismatch errors.
    sizes = [75, 76, 77, 78, 79]
    
    for size in sizes:
        # Create a tensor of the current size
        input_tensor = torch.randn(size)
        
        # Ensure at least one case triggers the 'else' branch (not all > 0)
        if size == 78:
            input_tensor[0] = -1.0

        # Run eager mode
        expected_output = func_with_all(input_tensor)
        
        # Run compiled mode
        actual_output = compiled_func(input_tensor)
        
        # Verify correctness
        assert torch.allclose(actual_output, expected_output), \
            f"Output mismatch for torch.all in compiled function at size {size}"

    print("Test passed: torch.all works correctly with torch.compile on dynamic shapes.")

if __name__ == "__main__":
    test_torch_all_compile_regression()