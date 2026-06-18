import torch
import tensorflow as tf

# Adapted from PyTorch MPS Buffer Allocation Regression Bug (Issue 167154)
# Original API: torch.nn.functional.linear
# Target API: tf.math.sobol_sample
#
# The original bug involved the MPS backend failing to allocate buffers correctly
# for non-contiguous tensors created via torch.as_strided.
# Since tf.math.sobol_sample generates data rather than accepting an input tensor,
# we cannot directly replicate the non-contiguous input scenario.
# However, we adapt the test to verify buffer allocation and execution
# using similar dimensions and data types to stress the backend.

def test_tf_sobol_sample():
    # Dimensions derived from the PyTorch example
    # PyTorch weight shape: (768, 768) -> dim = 768
    # PyTorch input shape: (5, 499, 768) -> num_results = 5 * 499 = 2495
    dim = 768
    num_results = 5 * 499

    # Use float32 as in the original bug report
    dtype = tf.float32

    # Attempt to run on GPU/MPS if available to test backend allocation
    device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

    with tf.device(device_name):
        # Call the similar API
        # Note: 'skip' is analogous to 'storage_offset' in terms of shifting the sequence start,
        # though the semantics differ (index shift vs memory offset).
        samples = tf.math.sobol_sample(
            dim=dim,
            num_results=num_results,
            skip=0,
            dtype=dtype
        )

        # Verify the operation completed and output shape is correct
        assert samples.shape == (num_results, dim), \
            f"Shape mismatch: expected ({num_results}, {dim}), got {samples.shape}"

        # Verify data type
        assert samples.dtype == dtype, \
            f"Dtype mismatch: expected {dtype}, got {samples.dtype}"

        # Verify values are within the expected Sobol range [0, 1)
        # This ensures the buffer was allocated and filled correctly
        assert tf.reduce_all(samples >= 0).numpy(), "Sobol samples contain negative values"
        assert tf.reduce_all(samples < 1.0).numpy(), "Sobol samples contain values >= 1.0"

        print("Test passed: tf.math.sobol_sample executed successfully with large dimensions.")

if __name__ == "__main__":
    test_tf_sobol_sample()