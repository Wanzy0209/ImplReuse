import torch
import torch.nn as nn

# Reproduce the configuration settings from the original bug report
# to ensure the test runs under the same compiler conditions.
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

def test_softmax_slicing():
    """
    Test case to verify that torch.compile handles slicing correctly
    when using the torch.nn.Softmax API, similar to the pattern
    reported in the issue with nonzero.
    """
    # Define the function using the similar API (Softmax)
    # and the slicing logic from the bug report.
    def f(x):
        # Leverage torch.nn.Softmax as the candidate for reuse
        softmax = nn.Softmax(dim=1)
        out = softmax(x)
        # Preserve the original bug reproduction logic: slicing the output
        return out[:-1]

    # Compile the function with fullgraph=True as in the original issue
    compiled_f = torch.compile(f, fullgraph=True)

    # Create a dummy input tensor
    input_tensor = torch.randn(3, 4)

    # Run the compiled function
    try:
        output = compiled_f(input_tensor)
        
        # Basic assertions to verify correctness
        assert output is not None, "Output should not be None"
        # Input shape (3, 4) -> Softmax (3, 4) -> Slice [:-1] -> (2, 4)
        assert output.shape == (2, 4), f"Expected shape (2, 4), got {output.shape}"
        
        print("Test passed successfully.")
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    test_softmax_slicing()