import tensorflow as tf
import numpy as np

def test_logical_or_tensor_slices():
    """
    Adapted from PyTorch Issue 161324.
    
    Original Bug: Data inconsistencies when using batch_isend_irecv with 2D tensor views.
    Adaptation: Verifying that tf.keras.ops.logical_or maintains data consistency 
    when operating on 2D tensor slices (views), ensuring no silent data corruption 
    occurs similar to the PyTorch distributed bug.
    """
    
    # Setup dimensions mimicking the original bug report
    batch_size = 4
    total_columns = 10
    
    # Define split offsets to create slices/views
    # PyTorch: split_offsets = [0, 5, 5, 10] (Splitting into two halves)
    split_offsets = [0, 5, 5, 10]
    
    # Create random boolean data (logical_or requires boolean inputs)
    # Using fixed seed for reproducibility
    np.random.seed(42)
    data_a = np.random.randint(0, 2, size=(batch_size, total_columns)).astype(bool)
    data_b = np.random.randint(0, 2, size=(batch_size, total_columns)).astype(bool)
    
    # Create TensorFlow tensors
    tensor_a = tf.constant(data_a)
    tensor_b = tf.constant(data_b)
    
    # --- Simulate the "View" creation ---
    # In PyTorch: dst_t1 = local_tensor[:, split_offsets[0]:split_offsets[1]]
    # In TensorFlow, slicing creates a new tensor, but we verify the operation 
    # on these subsets matches the operation on the whole.
    
    # Slice 1 (Columns 0 to 5)
    view_a1 = tensor_a[:, split_offsets[0]:split_offsets[1]]
    view_b1 = tensor_b[:, split_offsets[0]:split_offsets[1]]
    
    # Slice 2 (Columns 5 to 10)
    view_a2 = tensor_a[:, split_offsets[2]:split_offsets[3]]
    view_b2 = tensor_b[:, split_offsets[2]:split_offsets[3]]
    
    # --- Perform Operations on Views ---
    # Original PyTorch code used batch_isend_irecv. Here we perform the logical operation.
    # We check if the API handles the sliced inputs correctly without data corruption.
    result_view1 = tf.keras.ops.logical_or(view_a1, view_b1)
    result_view2 = tf.keras.ops.logical_or(view_a2, view_b2)
    
    # --- Calculate Ground Truth ---
    # Perform operation on the full tensor first, then slice.
    # If the API handles views correctly (or creates consistent new tensors),
    # these should match exactly.
    full_result = tf.keras.ops.logical_or(tensor_a, tensor_b)
    expected_view1 = full_result[:, split_offsets[0]:split_offsets[1]]
    expected_view2 = full_result[:, split_offsets[2]:split_offsets[3]]
    
    # --- Assertions ---
    # The original bug failed silently with inconsistent data. 
    # We explicitly check for equality here.
    try:
        assert tf.reduce_all(tf.equal(result_view1, expected_view1)).numpy(), \
            "Data inconsistency detected in view 1 (columns 0-5)"
        assert tf.reduce_all(tf.equal(result_view2, expected_view2)).numpy(), \
            "Data inconsistency detected in view 2 (columns 5-10)"
            
        print("Test Passed: tf.keras.ops.logical_or handles tensor slices consistently.")
        print(f"View 1 Shape: {result_view1.shape}, View 2 Shape: {result_view2.shape}")
        
    except AssertionError as e:
        print(f"Test Failed: {e}")
        raise

if __name__ == "__main__":
    test_logical_or_tensor_slices()