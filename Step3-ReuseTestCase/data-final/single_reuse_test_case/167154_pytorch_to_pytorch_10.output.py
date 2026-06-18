import torch

def test_randn_like_mps_non_contiguous():
    """
    Test case to verify torch.randn_like handles non-contiguous tensors 
    correctly on the MPS backend, similar to the buffer allocation issue 
    reported in Issue 167154.
    """
    if not torch.backends.mps.is_available():
        print("MPS device not available. Skipping test.")
        return

    # Setup: Create a non-contiguous tensor using the specific shape and stride
    # from the original bug report.
    shape = (5, 499, 768)
    stride = (0, 768, 1)
    storage_offset = 0
    
    # Calculate the number of elements required for the base storage
    numel = storage_offset + sum((shape[i] - 1) * stride[i] for i in range(len(shape))) + 1
    
    # Create the base tensor and the non-contiguous view
    base = torch.arange(numel, dtype=torch.float32, device="mps")
    input_tensor = torch.as_strided(base, size=shape, stride=stride, storage_offset=storage_offset)

    # Verify input_tensor is indeed non-contiguous
    assert not input_tensor.is_contiguous()

    # Test the similar API: torch.randn_like
    # This verifies that creating a new tensor based on the shape/layout of a 
    # non-contiguous tensor does not trigger MPS buffer allocation errors.
    try:
        output = torch.randn_like(input_tensor)
        
        # Assertions to verify the output properties match the input
        assert output.shape == input_tensor.shape, f"Shape mismatch: {output.shape} != {input_tensor.shape}"
        assert output.device == input_tensor.device, f"Device mismatch: {output.device} != {input_tensor.device}"
        assert output.dtype == input_tensor.dtype, f"Dtype mismatch: {output.dtype} != {input_tensor.dtype}"
        
        print("Test passed: torch.randn_like handled non-contiguous MPS tensor correctly.")
        
    except RuntimeError as e:
        if "buffer is not large enough" in str(e):
            print(f"Test failed: MPS buffer allocation error detected in torch.randn_like.\nError: {e}")
            raise
        else:
            raise

if __name__ == "__main__":
    test_randn_like_mps_non_contiguous()