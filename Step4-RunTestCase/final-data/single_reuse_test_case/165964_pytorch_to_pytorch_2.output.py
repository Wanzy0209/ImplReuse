import torch

# Test case adapted from Issue 165964 (torch.ones OOM on RTX 5090)
# Verifies if torch.eye encounters similar CUDA memory allocation issues.

if torch.cuda.is_available():
    try:
        # torch.eye(1) creates a 1x1 tensor with 1 on the diagonal.
        # This is semantically similar to torch.ones(1) in terms of memory allocation size.
        result = torch.eye(1, device="cuda")
        
        # Assertion to verify the tensor was created and contains the expected value
        assert result.shape == (1, 1), f"Expected shape (1, 1), got {result.shape}"
        assert result.item() == 1.0, f"Expected value 1.0, got {result.item()}"
        
        print("Test passed: torch.eye(1, device='cuda') executed successfully.")
    except RuntimeError as e:
        # Catching the specific error mentioned in the bug report (CUDA error: out of memory)
        print(f"Test failed with error: {e}")
else:
    print("CUDA is not available, skipping test.")