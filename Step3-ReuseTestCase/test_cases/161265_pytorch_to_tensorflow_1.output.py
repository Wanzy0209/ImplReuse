import tensorflow as tf

# Note: On macOS, '/GPU:0' typically maps to the Metal (MPS) backend if supported.
# This test attempts to reproduce the 4GB+ buffer initialization issue.
try:
    with tf.device('/GPU:0'):
        # Create a tensor larger than 4GB.
        # tf.keras.backend.random_uniform defaults to float32 (4 bytes per element).
        # Shape (2, (1 << 30) + 5) results in approx 2.15 billion elements.
        # Total memory size: ~8.6 GB, which exceeds the 4GB threshold mentioned in the bug.
        shape = (2, (1 << 30) + 5)
        
        # We use minval=1.0 so that if the buffer is not correctly filled (bug),
        # the values might be 0.0 (uninitialized) or otherwise distinguishable from the range [1.0, 2.0).
        a = tf.keras.backend.random_uniform(shape, minval=1.0, maxval=2.0, dtype='float32')

        # Access elements near the end of the allocated memory to check for initialization issues.
        # This mirrors the original bug report's check of a[1, -2] and a[:, -2].
        val_scalar = a[1, -2]
        val_slice = a[:, -2]

        print("Value at [1, -2]:", val_scalar.numpy())
        print("Value at [:, -2]:", val_slice.numpy())

        # Assert that the values are within the expected random range.
        # If the bug exists (buffer not filled beyond 4GB), these might be 0.0.
        assert val_scalar >= 1.0, f"Expected value >= 1.0, got {val_scalar}. Buffer might be uninitialized."
        assert tf.reduce_all(val_slice >= 1.0), f"Expected values >= 1.0, got {val_slice}. Buffer might be uninitialized."

        print("Test passed: Large tensor buffer appears correctly initialized.")

except RuntimeError as e:
    # Handle cases where GPU/MPS is not available or memory allocation fails.
    print(f"Test skipped or failed due to device/runtime constraints: {e}")