import tensorflow as tf
import numpy as np

def test_linear_operator_low_rank_update_batched_slices():
    """
    Test case for tf.linalg.LinearOperatorLowRankUpdate inspired by 
    PyTorch distributed issue #161324.
    
    The original issue involves data inconsistencies when transferring data 
    into specific slices (views) of a 2D tensor in a batched manner.
    
    This test adapts that logic to TensorFlow's LinearOperatorLowRankUpdate
    by verifying that a batched low-rank update correctly modifies specific
    columns (slices) of a base operator, ensuring data consistency.
    """
    
    # Setup dimensions mimicking the bug report's batch_size and total_columns
    batch_size = 4
    total_rows = 10
    total_columns = 20
    
    # Define two disjoint column ranges (slices) similar to the bug report's split_offsets
    # Original: dst_t1 = local_tensor[:, split_offsets[0]:split_offsets[1]]
    #           dst_t2 = local_tensor[:, split_offsets[2]:split_offsets[3]]
    slice1_start, slice1_end = 2, 7
    slice2_start, slice2_end = 12, 17
    
    k1 = slice1_end - slice1_start
    k2 = slice2_end - slice2_start
    total_k = k1 + k2
    
    # Create batched data for the updates (mimicking the tensors being sent)
    # Shape: [batch_size, total_rows, k]
    data1 = tf.random.normal([batch_size, total_rows, k1], seed=42)
    data2 = tf.random.normal([batch_size, total_rows, k2], seed=43)
    
    # Combine data for the U matrix in the update A = L + U V^H
    # U shape: [batch_size, total_rows, total_k]
    U = tf.concat([data1, data2], axis=2)
    
    # Create V (Selector matrix) to map the update data to specific columns
    # We want V such that U V^H places data1 into cols [2:7] and data2 into cols [12:17]
    # V is [total_columns, total_k]
    
    # Indices for the columns to be updated
    indices1 = tf.range(slice1_start, slice1_end)
    indices2 = tf.range(slice2_start, slice2_end)
    all_indices = tf.concat([indices1, indices2], axis=0)
    
    # Create one-hot encoding for V
    # V[col_idx, k_idx] = 1 if col_idx corresponds to the k-th update column
    V = tf.one_hot(all_indices, depth=total_columns, axis=0) # Shape [total_columns, total_k]
    
    # Base operator L (Zeros)
    # In the bug report, the destination tensor is initialized to zeros
    L = tf.linalg.LinearOperatorZeros(shape=[total_rows, total_columns], batch_shape=[batch_size])
    
    # Create the updated operator
    # A = L + U V^H
    # Since L is zeros, A = U V^H
    op = tf.linalg.LinearOperatorLowRankUpdate(L, U, V)
    
    # Compute the dense result
    result_dense = op.to_dense()
    
    # --- Verification ---
    
    # 1. Verify that the first slice contains data1
    res_slice1 = result_dense[:, :, slice1_start:slice1_end]
    tf.debugging.assert_near(res_slice1, data1, message="Data inconsistency in slice 1")
    
    # 2. Verify that the second slice contains data2
    res_slice2 = result_dense[:, :, slice2_start:slice2_end]
    tf.debugging.assert_near(res_slice2, data2, message="Data inconsistency in slice 2")
    
    # 3. Verify that other columns remain zero (checking for silent corruption)
    # Create a mask of columns that should be zero
    mask = tf.ones([total_columns], dtype=tf.bool)
    mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(all_indices, 1), tf.zeros_like(all_indices, dtype=tf.bool))
    
    # Extract columns that should be zero
    # We gather the columns and check if they are close to zero
    zero_cols = tf.boolean_mask(result_dense, mask, axis=2)
    tf.debugging.assert_near(zero_cols, tf.zeros_like(zero_cols), message="Non-zero data found in untouched columns")

    print("Test passed: Batched low-rank updates to tensor slices are consistent.")

if __name__ == "__main__":
    test_linear_operator_low_rank_update_batched_slices()