import torch

"""
Test torch.unique with the specific expression context that caused issues in FloorDiv.
The original bug involved FloorDiv being simplified to a Mul with a Rational in the 
symbolic engine. This test verifies that torch.unique behaves correctly with 
complex integer inputs, particularly under torch.compile which utilizes the 
symbolic engine.
"""

def test_torch_unique_complex_integers():
    print("Testing torch.unique with complex integer expression context...")

    # Create integer values based on the symbolic variables from the bug report
    # s14, s37, s46 were integer symbols
    s14_val = 4032
    s37_val = 10
    s46_val = 2

    # Build the input tensor data using the expression from the bug report
    # Original expression: (24*s37 + 672)*(((s14*s46)//2016)) + 21
    # We use this to generate the values in our tensor.
    
    # Step 1: Inner expression
    inner_val = (s14_val * s46_val) // 2016
    
    # Step 2: Middle expression
    middle_val = (24 * s37_val + 672) * inner_val
    
    # Step 3: Numerator
    data_val = middle_val + 21
    
    print(f"Generated data value from expression: {data_val}")

    # Create a tensor with duplicates to test unique
    # We add some variations to ensure unique has work to do
    input_tensor = torch.tensor([data_val, data_val, data_val + 1, data_val + 1, data_val + 2], dtype=torch.int64)
    
    print(f"Input tensor: {input_tensor}")

    # Define the function to test
    def unique_fn(x):
        return torch.unique(x)

    # Fix: Handle environments where torch.compile is not available (PyTorch < 2.0)
    if not hasattr(torch, 'compile'):
        print("Warning: torch.compile is not available in this environment. Skipping compilation and running in eager mode.")
        compiled_unique_fn = unique_fn
    else:
        # Compile the function to exercise the symbolic engine (torch._dynamo)
        # This is where the FloorDiv bug would have manifested.
        compiled_unique_fn = torch.compile(unique_fn)
    
    # Run the compiled function
    result = compiled_unique_fn(input_tensor)
    
    # Run the eager function for comparison
    expected = torch.unique(input_tensor)
    
    print(f"Result: {result}")
    print(f"Expected: {expected}")

    # Assert that the compiled result matches the eager result
    assert torch.equal(result, expected), f"Mismatch! Expected {expected}, got {result}"
    assert result.dtype == expected.dtype, f"Dtype mismatch! Expected {expected.dtype}, got {result.dtype}"

    print("Test passed: torch.unique works correctly with complex integer inputs under torch.compile.")

if __name__ == "__main__":
    test_torch_unique_complex_integers()