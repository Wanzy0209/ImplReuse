import tensorflow as tf
import numpy as np
import pytest

def test_erosion2d_device_compatibility():
    """
    Adapted test case based on Issue 160237 (aten::grid_sampler_3d not implemented on MPS).
    
    This test verifies the behavior of the similar API 'tf.nn.erosion2d' on the available 
    hardware (GPU/CPU). The original bug was a NotImplementedError when running a spatial 
    operation on a specific hardware backend (MPS). This test checks if tf.nn.erosion2d 
    executes successfully on the current TensorFlow device.
    """
    
    # Check for GPU availability (analogous to MPS availability in the bug report)
    gpus = tf.config.list_physical_devices('GPU')
    device_name = '/GPU:0' if gpus else '/CPU:0'
    
    print(f"Running test on device: {device_name}")

    with tf.device(device_name):
        # Create input tensor mimicking the batch/image structure found in the stack trace
        # Shape: [batch, in_height, in_width, depth]
        # Using float32 as is standard for such operations
        input_tensor = tf.constant(
            np.random.rand(1, 10, 10, 3).astype(np.float32),
            dtype=tf.float32
        )

        # Create kernel (structuring element)
        # Shape: [kernel_height, kernel_width, depth]
        kernel = tf.constant(
            np.ones((3, 3, 3), dtype=np.float32),
            dtype=tf.float32
        )

        # Define strides, rates, and padding
        strides = [1, 1, 1, 1]
        rates = [1, 1, 1, 1]
        padding = 'SAME' # Commonly used padding in image processing pipelines

        try:
            # Execute the operation
            # The original bug occurred during the execution of grid_sample
            output = tf.nn.erosion2d(
                value=input_tensor,
                kernel=kernel,
                strides=strides,
                rates=rates,
                padding=padding
            )

            # Verify that the operation produced a valid output tensor
            assert output is not None
            assert output.shape == input_tensor.shape # With 'SAME' padding, shape is preserved
            
            print(f"Operation successful on {device_name}. Output shape: {output.shape}")

        except NotImplementedError as e:
            # Replicating the error type from the original bug report
            pytest.fail(f"NotImplementedError encountered on {device_name}: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error on {device_name}: {e}")

if __name__ == "__main__":
    test_erosion2d_device_compatibility()