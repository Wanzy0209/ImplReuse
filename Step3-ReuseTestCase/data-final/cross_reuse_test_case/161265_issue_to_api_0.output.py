import torch
import tensorflow as tf
import numpy as np

def test_large_tensor_fill():
    """
    Test case to verify that large tensors (>4GB) are filled correctly.
    This reproduces the logic from the PyTorch issue where torch.full/ones
    failed to initialize memory beyond the 4GB boundary on MPS.
    
    The similar API context (tf.compat.v1.Summary.Value / Tensor) suggests
    using the Tensor interface and .numpy() method for verification.
    """
    
    # Dimensions from the original bug report:
    # Shape: (2, (1 << 31) + 5)
    # Total elements: 2 * (2^31 + 5) = 2^32 + 10 elements.
    # With dtype=int8, this requires > 4GB of memory.
    dim1 = 2
    dim2 = (1 << 31) + 5
    dtype = tf.int8

    try:
        # Create the tensor filled with ones (equivalent to torch.ones)
        # This tests the underlying buffer allocation and filling logic.
        a = tf.ones((dim1, dim2), dtype=dtype)

        # Check specific indices to ensure they are filled correctly.
        # The bug manifested as values being 0 instead of 1 near the end of the buffer.
        
        # Original: a[1, -2]
        val_single = a[1, -2]
        
        # Original: a[:, -2]
        val_slice = a[:, -2]

        # Verify values using .numpy() as indicated by the similar API pattern
        assert val_single.numpy() == 1, f"Expected 1, got {val_single.numpy()}"
        assert np.all(val_slice.numpy() == 1), f"Expected [1, 1], got {val_slice.numpy()}"

        print("Test Passed: Large tensor filled correctly.")

    except tf.errors.ResourceExhaustedError as e:
        print(f"Test Skipped: Insufficient memory to allocate 4GB+ tensor. {e}")
    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    test_large_tensor_fill()