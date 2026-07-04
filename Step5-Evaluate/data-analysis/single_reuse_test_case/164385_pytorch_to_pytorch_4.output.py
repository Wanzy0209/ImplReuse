import torch

def test_torch_square_complex():
    """
    Test torch.square with a complex symbolic expression similar to the FloorDiv issue.
    The original issue involved FloorDiv being simplified to a Rational in the symbolic engine.
    This test ensures torch.square behaves correctly in a similar complex arithmetic context
    when compiled.
    """
    # Check if torch.compile is available (requires PyTorch >= 2.0)
    if not hasattr(torch, 'compile'):
        print("Skipping test: torch.compile is not available in this PyTorch version (requires PyTorch >= 2.0).")
        return

    # Create input tensors
    # Using integers to match the context of the original bug report
    s14 = torch.tensor(4032, dtype=torch.int32)
    s37 = torch.tensor(1, dtype=torch.int32)
    s46 = torch.tensor(1, dtype=torch.int32)

    print("Testing torch.square with complex symbolic expression...")

    # Define the model
    def model(s14, s37, s46):
        # Build the expression step by step, mirroring the bug report's structure
        # Original: inner_expr = FloorDiv(s14*s46 , 2016)
        inner_expr = torch.div(s14 * s46, 2016, rounding_mode='floor')
        
        # Original: middle_expr = (24 * s37 + 672) * inner_expr
        middle_expr = (24 * s37 + 672) * inner_expr
        
        # Original: numerator = middle_expr + 21
        numerator = middle_expr + 21
        
        # Original API: FloorDiv(numerator, 22)
        # Similar API: torch.square(numerator)
        result = torch.square(numerator)
        return result

    # Compile the model to engage the symbolic engine (torch._inductor/torch._dynamo)
    # This is where the FloorDiv -> Rational bug occurred.
    compiled_model = torch.compile(model)
    
    # Execute
    result = compiled_model(s14, s37, s46)
    expected = model(s14, s37, s46)

    print(f"Result: {result}")
    print(f"Expected: {expected}")

    # Verify
    assert torch.equal(result, expected), f"Test failed: {result} != {expected}"
    print("Test passed successfully.")

if __name__ == "__main__":
    test_torch_square_complex()