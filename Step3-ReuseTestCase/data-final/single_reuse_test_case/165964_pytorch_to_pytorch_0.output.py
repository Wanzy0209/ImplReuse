import torch
import sys

def test_torch_ones_cuda():
    """
    Test case to verify torch.ones works on CUDA without out-of-memory errors.
    Based on Issue ID: 165964
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    try:
        # Original failing call: torch.ones(1, device="cuda")
        # Similar API: torch.ones
        tensor = torch.ones(1, device="cuda")
        
        # Verify the tensor is on the correct device
        assert tensor.is_cuda, "Tensor is not on CUDA device"
        
        # Verify the value is correct
        value = tensor.item()
        assert value == 1.0, f"Expected value 1.0, but got {value}"
        
        print("Test passed: torch.ones(1, device='cuda') executed successfully.")
        
    except torch.AcceleratorError as e:
        print(f"Test failed with CUDA error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_torch_ones_cuda()