import tensorflow as tf
import numpy as np
import sys

def test_ragged_stack_dynamic_partitions():
    """
    Test case adapted from the PyTorch 'aten::grid_sampler_3d' issue (Issue 160237).
    
    The original bug involved a NotImplementedError for grid sampling operations on 
    the MPS device within a feature warping pipeline. This test verifies the 
    TensorFlow equivalent operation, `tf.ragged.stack_dynamic_partitions`, ensuring 
    it correctly handles the gathering and stacking of tensor data based on 
    partition indices (semantically similar to sampling/gathering features).
    """
    
    # Setup: Simulating the 'feature_3d' and grid-like partitioning logic
    # from the original PyTorch stack trace (warping_network.py).
    # We use float data to resemble the feature tensors typically processed in such pipelines.
    input_data = tf.constant([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=tf.float32)
    
    # Partitions act similarly to grid coordinates, determining where each data point belongs.
    # In the original bug, the grid defined sampling locations; here, partitions define stacking groups.
    partitions = tf.constant([0, 2, 0, 1, 2, 1], dtype=tf.int32)
    num_partitions = 3

    # Execution: Attempt to perform the operation.
    # The original bug failed on MPS. Here we verify the TF op works on the available device.
    try:
        # Explicitly trying to use GPU if available to mimic the device-specific nature of the original bug.
        # If not available, it falls back to CPU automatically.
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            result = tf.ragged.stack_dynamic_partitions(input_data, partitions, num_partitions)
    except Exception as e:
        print(f"Test failed with exception: {e}")
        sys.exit(1)

    # Verification: Check if the output is a RaggedTensor and contains the correct values.
    assert isinstance(result, tf.RaggedTensor), "Output should be a RaggedTensor"
    
    # Expected logic:
    # Partition 0: indices 0, 2 -> values 10.0, 30.0
    # Partition 1: indices 3, 5 -> values 40.0, 60.0
    # Partition 2: indices 1, 4 -> values 20.0, 50.0
    
    expected_values = [
        [10.0, 30.0],
        [40.0, 60.0],
        [20.0, 50.0]
    ]

    # Assert row by row to handle RaggedTensor structure
    for i in range(num_partitions):
        np.testing.assert_array_almost_equal(
            result[i].numpy(), 
            expected_values[i], 
            decimal=5,
            err_msg=f"Mismatch in partition {i}"
        )

    print("Test passed: tf.ragged.stack_dynamic_partitions executed successfully.")

if __name__ == "__main__":
    test_ragged_stack_dynamic_partitions()