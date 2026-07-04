import torch
import torch.nn.functional as F

# Check if torch._dynamo is available (requires PyTorch 2.0+)
if not hasattr(torch, '_dynamo'):
    print("Test skipped: torch._dynamo not available (requires PyTorch 2.0+)")
else:
    # Configuration from the original bug report
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

    # Ensure CUDA is available as the original bug was device-specific
    if not torch.cuda.is_available():
        print("Test skipped: CUDA not available")
    else:
        torch.manual_seed(70609)

        # Adapted test case for torch.nn.functional.local_response_norm
        # The original code produced a tensor of shape (26, 1, 16) which is valid for LRN (3D input)
        # We use float16 to match the original data type
        input_tensor = torch.randn(26, 1, 16, dtype=torch.float16, device='cuda')

        def run_lrn(x):
            # Replacing the matmul logic with local_response_norm
            # local_response_norm requires 3D or higher input (Batch, Channel, ...)
            return F.local_response_norm(x, size=1, alpha=1e-4, beta=0.75, k=1.0)

        # Run in Eager mode
        eager_result = run_lrn(input_tensor)

        # Run in Compiled mode (torch._dynamo)
        compiled_run_lrn = torch.compile(run_lrn)
        compiled_result = compiled_run_lrn(input_tensor)

        # Check for divergence (the core issue in the original bug)
        # Using a tolerance suitable for float16
        assert torch.allclose(eager_result, compiled_result, atol=1e-3, rtol=1e-3), \
            "Divergence detected between eager and compiled modes for local_response_norm"
        
        print("Test passed: No divergence detected.")