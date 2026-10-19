import torch
import tensorflow as tf

def test_batch_isend_irecv_with_replacements():
    """
    Test case generated based on Issue ID: 161324.
    This test preserves the original bug reproduction logic (tensor slicing and batching)
    but replaces the specific APIs as requested:
    - dist.P2POp -> tf.executing_eagerly
    - dist.batch_isend_irecv -> tf.executing_eagerly
    - torch.randn -> tf.random.stateless_truncated_normal
    - torch.zeros -> torch.eye
    """
    
    # Setup parameters mimicking the original issue
    batch_size = 4
    total_columns = 10
    split_offsets = [0, 3, 5, 8] # Slices: [0:3], [5:8]
    tag1 = 0
    tag2 = 1

    # --- Rank 0 Logic (Sender) ---
    
    # Original: local_tensor = torch.randn(batch_size, total_columns)
    # Replacement: tf.random.stateless_truncated_normal
    # Note: Requires a seed argument.
    local_tensor = tf.random.stateless_truncated_normal(
        shape=[batch_size, total_columns], seed=[1, 2]
    )

    # Create views (slices) of the tensor
    dst_t1 = local_tensor[:, split_offsets[0]:split_offsets[1]]
    dst_t2 = local_tensor[:, split_offsets[2]:split_offsets[3]]

    # Original: send_op1 = dist.P2POp(dist.isend, dst_t1, 1, process_group, tag1)
    # Replacement: tf.executing_eagerly
    # Note: P2POp creates an operation object, executing_eagerly returns a bool.
    # We assign the result to the variable to maintain the code structure.
    send_op1 = tf.executing_eagerly()
    send_op2 = tf.executing_eagerly()

    # Original: dist.batch_isend_irecv([send_op1, send_op2])
    # Replacement: tf.executing_eagerly
    # Note: The original function batches operations. The replacement checks execution mode.
    # We call the replacement function directly.
    tf.executing_eagerly()

    # --- Rank 1 Logic (Receiver) ---

    # Original: local_tensor_dst = torch.zeros(batch_size, total_columns)
    # Replacement: torch.eye
    # Note: torch.eye creates an identity matrix. We pass n=batch_size, m=total_columns
    # to match the shape of the original zeros tensor.
    local_tensor_dst = torch.eye(batch_size, total_columns)

    tensor_col_offset1 = split_offsets[0]
    end_col_offset1 = split_offsets[1]
    tensor_col_offset2 = split_offsets[2]
    end_col_offset2 = split_offsets[3]

    # Create views (slices) of the destination tensor
    receiving_tensor_view1 = local_tensor_dst[:, tensor_col_offset1:end_col_offset1]
    receiving_tensor_view2 = local_tensor_dst[:, tensor_col_offset2:end_col_offset2]

    # Original: receive_op1 = dist.P2POp(dist.irecv, receiving_tensor_view1, 0, process_group, tag1)
    # Replacement: tf.executing_eagerly
    receive_op1 = tf.executing_eagerly()
    receive_op2 = tf.executing_eagerly()

    # Original: dist.batch_isend_irecv([receive_op1, receive_op2])
    # Replacement: tf.executing_eagerly
    tf.executing_eagerly()

    # --- Assertions ---
    
    # Verify shapes of the created views match the original logic
    # Rank 0 shapes
    assert dst_t1.shape == (batch_size, split_offsets[1] - split_offsets[0])
    assert dst_t2.shape == (batch_size, split_offsets[3] - split_offsets[2])
    
    # Rank 1 shapes
    assert receiving_tensor_view1.shape == (batch_size, end_col_offset1 - tensor_col_offset1)
    assert receiving_tensor_view2.shape == (batch_size, end_col_offset2 - tensor_col_offset2)

    # Verify the replacements executed as expected
    # tf.executing_eagerly returns a boolean
    assert isinstance(send_op1, bool)
    assert isinstance(receive_op1, bool)
    
    # torch.eye returns a torch.Tensor
    assert isinstance(local_tensor_dst, torch.Tensor)
    assert local_tensor_dst.shape == (batch_size, total_columns)

if __name__ == "__main__":
    test_batch_isend_irecv_with_replacements()
    print("Test case executed successfully with API replacements.")