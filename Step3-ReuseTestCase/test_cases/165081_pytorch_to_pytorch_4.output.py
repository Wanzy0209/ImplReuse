import torch
import torch._dynamo

# Reproduce the configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch.manual_seed(52676)

# Check for CUDA availability as the original bug was device-specific
if torch.cuda.is_available():
    device = 'cuda'
    dtype = torch.float64

    # Create inputs suitable for torch.mv (Matrix x Vector)
    # Using dimensions similar to the bug report (e.g., 9)
    matrix = torch.randn(9, 9, dtype=dtype, device=device)
    vector = torch.randn(9, dtype=dtype, device=device)

    def test_function(mat, vec):
        # Adapted call site: using torch.mv instead of the original matmul/nonzero chain
        return torch.mv(mat, vec)

    # Run eager mode
    expected = test_function(matrix, vector)

    # Run compiled mode
    compiled_fn = torch._dynamo.optimize(test_function)
    actual = compiled_fn(matrix, vector)

    # Assertion to verify behavior matches between eager and compiled modes
    assert torch.allclose(expected, actual), "Eager and compiled results differ"
    print("Test passed successfully.")
else:
    print("CUDA device not found. Skipping test.")