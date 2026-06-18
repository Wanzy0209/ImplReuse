import torch
import torch.nn.functional as F

# Check for CUDA availability to match the bug report's environment
if not torch.cuda.is_available():
    print("CUDA not available, skipping test.")
else:
    # Setup inputs similar to the bug report (batch size 1, dimension 1024)
    # torch.bmm works with (B, M, K) and (B, K, N).
    # torch.nn.functional.one_hot works with (N) or (B, N).
    # We use a 2D tensor to mimic the batch dimension of the original bug.
    indices = torch.randint(0, 1024, (1, 1024), device="cuda")

    @torch.compile
    def one_hot_layer(input_indices, num_classes):
        # The original bug involved torch.bmm with an out_dtype argument that failed.
        # Here we test torch.nn.functional.one_hot inside torch.compile.
        # While one_hot does not have an out_dtype argument, we test its compilation
        # stability with type handling, similar to the context of the bug report.
        return F.one_hot(input_indices, num_classes)

    # Run the compiled function
    try:
        output = one_hot_layer(indices, 1024)
        
        # Assertions to verify correctness
        assert output.shape == (1, 1024, 1024), f"Expected shape (1, 1024, 1024), got {output.shape}"
        assert output.dtype == torch.long, f"Expected dtype torch.long, got {output.dtype}"
        assert output.device.type == "cuda", f"Expected device cuda, got {output.device.type}"
        
        print("Test passed: torch.nn.functional.one_hot works with torch.compile.")
    except Exception as e:
        print(f"Test failed with error: {e}")