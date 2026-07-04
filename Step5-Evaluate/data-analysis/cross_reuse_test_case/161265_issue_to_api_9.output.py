import torch

def test_torch_full_large_tensor_mps():
    # Leverage the pattern from the similar API (torch.backends.cuda.is_built)
    # to check for backend availability before running the test.
    if not torch.backends.mps.is_available():
        print("Skipping test: MPS backend is not available.")
        return

    # Reproduce the bug: torch.full fails for tensors > 4GB on MacOS MPS.
    # We create a tensor slightly larger than 4GB.
    # Size calculation: 2 * (2^31 + 5) elements * 1 byte (int8) ~= 4GB + 10 bytes.
    rows = 2
    cols = (1 << 31) + 5
    fill_value = 1

    # Use torch.full directly as indicated by the Issue Title.
    # Note: The original bug report used torch.ones, which is a wrapper around torch.full.
    tensor = torch.full((rows, cols), fill_value, dtype=torch.int8, device='mps')

    # Verify the tail elements where the fillBuffer logic failed.
    # The bug report showed these elements returning 0 instead of 1.
    assert tensor[1, -2].item() == fill_value, \
        f"Expected tail element to be {fill_value}, got {tensor[1, -2].item()}"
    
    assert tensor[0, -2].item() == fill_value, \
        f"Expected tail element to be {fill_value}, got {tensor[0, -2].item()}"

    # Verify the slice to ensure consistency across the boundary
    expected_slice = torch.tensor([fill_value, fill_value], dtype=torch.int8, device='mps')
    assert torch.equal(tensor[:, -2], expected_slice), \
        "Slice at the end of the buffer does not match the fill value"

    print("Test passed: torch.full correctly fills tensors larger than 4GB on MPS.")

if __name__ == "__main__":
    test_torch_full_large_tensor_mps()