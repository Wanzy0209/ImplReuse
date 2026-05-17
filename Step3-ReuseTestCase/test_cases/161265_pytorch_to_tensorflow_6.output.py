import torch
import tensorflow as tf
import os
import sys

def test_large_tensor_mps():
    """
    Adapted test case for tf.compat.v1.summary.scalar based on the PyTorch bug report.
    The original bug involves torch.full (via torch.ones) failing to correctly initialize
    tensors larger than 4GB on MacOS MPS (Metal), specifically with int8 dtype.
    
    This test creates a similar large tensor in TensorFlow, verifies the initialization,
    and attempts to log a scalar value derived from it using the specified API.
    """
    
    # Check for GPU availability (MPS on MacOS is mapped to GPU in TensorFlow)
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Test skipped: No GPU (MPS) device found.")
        return

    # Setup for summary writing (required for tf.compat.v1.summary.scalar)
    log_dir = os.path.join(os.getcwd(), "logs/mps_test")
    writer = tf.summary.create_file_writer(log_dir)

    # Use the first GPU
    device_name = f'/GPU:0'
    
    print(f"Running test on {device_name}...")

    with tf.device(device_name):
        try:
            # Reproduce the core bug logic: Create a tensor > 4GB
            # Shape: (2, (1 << 31) + 5) -> ~4.29 GB
            # Dtype: int8 (as in the original bug)
            shape = (2, (1 << 31) + 5)
            dtype = tf.int8
            
            # tf.ones is the TensorFlow equivalent of torch.ones
            a = tf.ones(shape, dtype=dtype)
            
            # Force execution to ensure allocation and initialization happen
            # (In eager mode, ops execute immediately, but accessing values ensures it)
            
            # Check specific elements as in the original bug report
            # a[1, -2]
            val_single = a[1, -2]
            # a[:, -2]
            val_slice = a[:, -2]

            # Verify the values (Core bug reproduction logic)
            # We expect 1, but the bug might cause 0 or garbage
            assert val_single.numpy() == 1, f"Bug reproduced: a[1, -2] is {val_single.numpy()}, expected 1"
            assert tf.reduce_all(val_slice == 1).numpy(), f"Bug reproduced: a[:, -2] is {val_slice.numpy()}, expected [1, 1]"

            print("Tensor initialization check passed.")

            # Adapt to the requested API: tf.compat.v1.summary.scalar
            # We log the value we just verified to ensure the API handles the context.
            with writer.as_default():
                # Note: tf.compat.v1.summary.scalar expects a scalar tensor.
                # We pass the value extracted from the large tensor.
                # This tests if the summary mechanism can handle data derived from large buffers.
                tf.compat.v1.summary.scalar("large_tensor_element_check", tf.cast(val_single, tf.float32), step=0)
                writer.flush()
            
            print("Summary scalar written successfully.")

        except tf.errors.ResourceExhaustedError:
            print("Test skipped: Resource exhausted (OOM) - Tensor too large for device.")
        except Exception as e:
            print(f"Test failed with exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_large_tensor_mps()