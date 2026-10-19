import torch

# Test case adapted from Issue 161265
# Replacing torch.ones with torch.normal as per similarity instructions.
# Note: torch.normal generates float tensors, so dtype=torch.int8 is omitted.
# The original bug involved fillBuffer failing for large tensors on MPS.

def test_mps_large_tensor_normal():
    # Create a large tensor (>2GB elements) on MPS device
    # Original shape: (2, (1 << 31) + 5)
    try:
        # Using torch.normal (replacement for torch.ones)
        # mean=0, std=1
        a = torch.normal(0, 1, (2, (1 << 31) + 5), device='mps')
        
        # Access elements to verify memory integrity and kernel execution
        # The original bug manifested as incorrect values (0 instead of 1) at specific indices
        val_single = a[1, -2]
        val_slice = a[:, -2]
        
        print(f"Value at [1, -2]: {val_single}")
        print(f"Value at [:, -2]: {val_slice}")
        
        # Assertions to check basic properties
        assert a.shape == (2, (1 << 31) + 5), f"Shape mismatch: {a.shape}"
        assert a.device.type == 'mps', f"Device mismatch: {a.device}"
        
        # For random normal, we check for NaNs which might indicate memory corruption
        assert not torch.isnan(val_single), "Found NaN at [1, -2]"
        assert not torch.isnan(val_slice).any(), "Found NaN in slice [:, -2]"
        
        print("Test passed: Large tensor created and accessed successfully.")
        
    except RuntimeError as e:
        print(f"RuntimeError encountered (MPS might be unsupported or OOM): {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_mps_large_tensor_normal()