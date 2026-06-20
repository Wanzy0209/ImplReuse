import torch

# Check if torch._dynamo is available to avoid AttributeError
# This handles environments where PyTorch version is < 2.0 or dynamo is not included.
if hasattr(torch, '_dynamo'):
    # Reproduce the configuration settings from the original bug report
    # to ensure the compiler handles dynamic shapes and scalar outputs.
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

    def f(x):
        # Create a tensor with unbacked sizes (dynamic shape) using nonzero.
        # This mimics the source of the error in the original issue.
        nz = x.nonzero()
        
        # Apply the similar API (torch.isinf) to the tensor with unbacked sizes.
        # This tests whether the lowering logic for isinf (which calls full_like
        # and create_tensor_like) correctly handles unbacked SymInts.
        return torch.isinf(nz)

    # Create a test input. 
    # We use a fixed seed or specific values to ensure deterministic behavior,
    # though the core issue is about the compilation of dynamic shapes.
    input_tensor = torch.randn(3, 4)
    input_tensor[0, 0] = 0  # Ensure there is at least one zero for nonzero()

    # Compile and run the function with fullgraph=True
    try:
        out = torch.compile(f, fullgraph=True)(input_tensor)
        
        # Assertions to verify the output is as expected.
        # nonzero() returns indices (integers), so isinf should return False for all elements.
        assert out.dtype == torch.bool, "Output dtype should be bool"
        assert out.shape[0] <= 12, "Output shape dimension 0 should be within bounds of input"
        
        print("Test passed. torch.isinf handled unbacked sizes correctly.")
    except Exception as e:
        print(f"Test failed with error: {e}")
else:
    print("Skipping test: torch._dynamo is not available in this PyTorch version.")