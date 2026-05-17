import tensorflow as tf
import numpy as np

def test_logical_and_2d_tensor_views():
    """
    Adapted test case to verify data consistency when using 
    tf.experimental.numpy.logical_and with 2D tensor views.
    
    The original PyTorch bug (Issue 161324) involved data inconsistencies 
    when transferring data using 2D tensor views. This test checks if 
    tf.experimental.numpy.logical_and produces consistent results when 
    operating on views of 2D tensors compared to operating on the full tensors.
    """
    
    # Setup dimensions mimicking the PyTorch scenario
    batch_size = 4
    total_columns = 10

    # Create two 2D tensors (mimicking the sender and receiver data contexts)
    # Using random data to ensure general coverage. 
    # Note: logical_and casts inputs to bool, so integers 0/1 or floats are fine.
    tensor_a = tf.random.uniform((batch_size, total_columns), minval=0, maxval=2, dtype=tf.int32)
    tensor_b = tf.random.uniform((batch_size, total_columns), minval=0, maxval=2, dtype=tf.int32)

    # Define offsets for slicing (mimicking split_offsets in the original bug)
    # PyTorch: split_offsets[0]:split_offsets[1] and split_offsets[2]:split_offsets[3]
    split_offsets = [0, 3, 6, 9]

    # Create views (slices) for the first operation
    # Corresponds to dst_t1 / receiving_tensor_view1 in the original bug
    view_a1 = tensor_a[:, split_offsets[0]:split_offsets[1]]
    view_b1 = tensor_b[:, split_offsets[0]:split_offsets[1]]

    # Create views (slices) for the second operation
    # Corresponds to dst_t2 / receiving_tensor_view2 in the original bug
    view_a2 = tensor_a[:, split_offsets[2]:split_offsets[3]]
    view_b2 = tensor_b[:, split_offsets[2]:split_offsets[3]]

    # Apply the API: tf.experimental.numpy.logical_and on the views
    # This mimics the batch_isend_irecv operations on views
    result_view1 = tf.experimental.numpy.logical_and(view_a1, view_b1)
    result_view2 = tf.experimental.numpy.logical_and(view_a2, view_b2)

    # Calculate expected result by operating on full tensors first
    # This checks for data consistency (the bug in PyTorch was inconsistency)
    full_result = tf.experimental.numpy.logical_and(tensor_a, tensor_b)

    # Slice the full result to get the ground truth
    expected_view1 = full_result[:, split_offsets[0]:split_offsets[1]]
    expected_view2 = full_result[:, split_offsets[2]:split_offsets[3]]

    # Assertions to verify data consistency
    # If the API has a similar bug regarding views, these assertions will fail
    assert tf.reduce_all(tf.equal(result_view1, expected_view1)).numpy(), \
        "Data inconsistency found in view 1 using tf.experimental.numpy.logical_and"
    
    assert tf.reduce_all(tf.equal(result_view2, expected_view2)).numpy(), \
        "Data inconsistency found in view 2 using tf.experimental.numpy.logical_and"

    print("Test passed: tf.experimental.numpy.logical_and handles 2D tensor views consistently.")

if __name__ == "__main__":
    test_logical_and_2d_tensor_views()