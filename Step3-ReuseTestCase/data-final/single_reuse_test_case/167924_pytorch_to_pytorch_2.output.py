import torch

# Check for MPS availability to ensure the test runs on the correct backend
if not torch.backends.mps.is_available():
    print("MPS device is not available. Skipping test.")
else:
    # Setup: Create a tensor on MPS using torch.arange (Original API Under Test)
    # We use float type as std typically operates on floating point inputs
    data = torch.arange(10, device="mps", dtype=torch.float)
    
    # Action: Create a non-prefix slice of the tensor
    # This mimics the 'counts[1:3]' pattern from the bug report which triggered the crash
    sliced_data = data[1:5]
    
    # Test: Call the similar API (torch.std) with the sliced tensor
    # Adapted from the original call site: data.repeat_interleave(counts[1:3], dim=0)
    # We verify that torch.std handles the sliced MPS tensor without crashing
    try:
        result = torch.std(sliced_data)
        
        # Assertion to verify the operation completes and returns a valid result
        assert result is not None
        assert not torch.isnan(result)
        print(f"Test passed. Result: {result}")
    except RuntimeError as e:
        print(f"Test failed with RuntimeError: {e}")
    except Exception as e:
        print(f"Test failed with unexpected exception: {e}")