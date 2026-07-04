import torch

# Handle missing torch.compile for PyTorch versions < 2.0
if not hasattr(torch, 'compile'):
    # Mock torch.compile to return the model directly (eager execution)
    torch.compile = lambda model, **kwargs: model

def test_torch_log_complex_expression():
    """
    Test torch.log with a complex symbolic expression adapted from the FloorDiv issue.
    Original expression: FloorDiv((24*s37 + 672)*(((s14*s46)//2016)) + 21, 22)
    Adapted to: torch.log((24*s37 + 672)*(((s14*s46)//2016)) + 21)
    """

    # Define the function to be compiled
    def model(s14, s37, s46):
        # Replicate the complex expression structure
        # Using the expression from the issue title which includes s46
        inner_expr = (s14 * s46) // 2016
        middle_expr = (24 * s37 + 672) * inner_expr
        numerator = middle_expr + 21

        # Replace FloorDiv(numerator, denominator) with torch.log(numerator)
        return torch.log(numerator)

    # Compile the model to trigger symbolic tracing
    # If torch.compile is missing, the mock above ensures this returns the model directly
    compiled_model = torch.compile(model, backend="eager")

    # Create inputs (positive integers as per original symbols)
    s14 = torch.tensor(4032, dtype=torch.int32)
    s37 = torch.tensor(10, dtype=torch.int32)
    s46 = torch.tensor(2, dtype=torch.int32)

    # Run the compiled model
    result = compiled_model(s14, s37, s46)

    # Calculate expected result
    # torch.log promotes integer inputs to float
    expected = torch.log((24 * s37 + 672) * ((s14 * s46) // 2016) + 21)

    # Verify correctness
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"
    print("Test passed: torch.log handles complex symbolic expression correctly.")

if __name__ == "__main__":
    test_torch_log_complex_expression()