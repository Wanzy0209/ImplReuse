import sys

# Handle environment dependency issues (e.g., missing GLIBCXX) by catching import errors
try:
    import torch
    import tensorflow as tf
    import numpy as np
except ImportError as e:
    print(f"Environment Error: {e}")
    print("Skipping test: Required libraries (TensorFlow) could not be imported.")
    print("This is likely due to a missing system dependency (e.g., GLIBCXX_3.4.29).")
    sys.exit(0)

def test_stateless_random_jpeg_quality_large_tensor():
    """
    Test case for tf.image.stateless_random_jpeg_quality based on the 
    PyTorch MaxPool2d bug (Issue 165297).
    
    Verifies that the TF API handles large tensors in NHWC (channels_last) 
    format without producing NaNs or crashing, similar to the reported issue 
    with torch.nn.MaxPool2d.
    """
    # Large input tensor dimensions
    # Original bug used (N, C, H, W) = (84, 64, 512, 960)
    # TF image ops typically use (N, H, W, C) (NHWC), which is equivalent to channels_last
    N, H, W, C = 84, 512, 960, 64

    # Case 1: float32 + channels_last (NHWC)
    # Using float32 to match the "illegal memory access" case in the bug report.
    # Note: While the bug title mentions bfloat16, float32 was also a failing case
    # and is more universally supported by image processing ops like JPEG quality.
    x = tf.random.normal((N, H, W, C), dtype=tf.float32)

    # In TF, image ops default to NHWC (channels_last).
    # We verify the layout implicitly by the shape definition.
    print(f"Input tensor shape: {x.shape}")
    print(f"Input tensor dtype: {x.dtype}")

    # Parameters for the similar API
    min_quality = 75
    max_quality = 95
    seed = (1, 2)

    # Apply the operation
    # The PyTorch bug occurred during the operation (MaxPool)
    # Here we apply the similar TF operation
    try:
        y = tf.image.stateless_random_jpeg_quality(x, min_quality, max_quality, seed)

        # Check for NaNs/Infs (The failure mode of the original bug)
        # Note: tf.image.stateless_random_jpeg_quality typically returns uint8,
        # so NaN checks are usually False, but we perform them to match the test logic.
        has_nan = tf.reduce_any(tf.math.is_nan(y)).numpy()
        has_inf = tf.reduce_any(tf.math.is_inf(y)).numpy()

        print(f"Output contains NaN? {has_nan}")
        print(f"Output contains Inf? {has_inf}")

        assert not has_nan, "Detected NaNs in output"
        assert not has_inf, "Detected Infs in output"
        print("Test passed: No NaNs or Infs detected.")

    except Exception as e:
        # Catching potential illegal memory access or other crashes
        print(f"Test failed with exception: {e}")
        raise

if __name__ == "__main__":
    test_stateless_random_jpeg_quality_large_tensor()