import torch
import torch.nn.functional as F
import tensorflow as tf

def test_batch_isend_irecv_with_replacements():
    """
    Test case generated based on Issue ID: 161324.
    Replaces specific APIs as requested:
    - dist.P2POp -> tf.custom_gradient
    - dist.batch_isend_irecv -> tf.custom_gradient
    - torch.randn -> torch.nn.functional.max_pool1d
    - torch.zeros -> torch.dist
    """
    
    # Setup parameters
    batch_size = 2
    total_columns = 10
    split_offsets = [0, 4, 4, 8]
    tag1 = 0
    tag2 = 1
    
    # Simulating rank 0 (Sender)
    # Replacement: torch.randn -> torch.nn.functional.max_pool1d
    # We need an input tensor for max_pool1d. Creating a dummy input.
    # max_pool1d expects (N, C, L) or (C, L). We'll use (1, batch_size, total_columns) 
    # and squeeze/reshape to match original (batch_size, total_columns).
    dummy_input_rank0 = torch.ones(1, batch_size, total_columns)
    local_tensor = F.max_pool1d(dummy_input_rank0, kernel_size=1).squeeze(0)
    
    dst_t1 = local_tensor[:, split_offsets[0]:split_offsets[1]]
    dst_t2 = local_tensor[:, split_offsets[2]:split_offsets[3]]
    
    # Replacement: dist.P2POp -> tf.custom_gradient
    # tf.custom_gradient is a decorator. We wrap the logic in a lambda/function 
    # to mimic the creation of an operation object.
    # We pass the arguments that P2POp would have taken (op, tensor, peer, pg, tag)
    # to the lambda to maintain signature structure, though tf.custom_gradient 
    # primarily takes a function `f`.
    send_op1 = tf.custom_gradient(lambda op, t, p, pg, tag: t)(None, dst_t1, 1, None, tag1)
    send_op2 = tf.custom_gradient(lambda op, t, p, pg, tag: t)(None, dst_t2, 1, None, tag2)
    
    # Replacement: dist.batch_isend_irecv -> tf.custom_gradient
    # We wrap the list of operations in a lambda to satisfy the decorator's requirement
    # for a callable, effectively defining a gradient function for the batch.
    batch_op = tf.custom_gradient(lambda ops: [op() if callable(op) else op for op in ops])([send_op1, send_op2])
    
    print(f"Rank 0 processed tensors with shapes {dst_t1.shape} and {dst_t2.shape}")

    # Simulating rank 1 (Receiver)
    # Replacement: torch.zeros -> torch.dist
    # torch.dist computes the p-norm distance between two tensors and returns a scalar.
    # To maintain runnability with the subsequent slicing logic (which expects a 2D tensor),
    # we compute the distance and then wrap it in a tensor of the required shape.
    # This preserves the API call while adapting the semantics to fit the test flow.
    dummy_t1 = torch.ones(batch_size, total_columns)
    dummy_t2 = torch.zeros(batch_size, total_columns)
    dist_val = torch.dist(dummy_t1, dummy_t2, p=2)
    
    # Wrapping scalar distance into the expected tensor shape to allow slicing
    local_tensor_dst = torch.full((batch_size, total_columns), dist_val)
    
    tensor_col_offset1 = split_offsets[0]
    end_col_offset1 = split_offsets[1]
    tensor_col_offset2 = split_offsets[2]
    end_col_offset2 = split_offsets[3]
    
    receiving_tensor_view1 = local_tensor_dst[:, tensor_col_offset1:end_col_offset1]
    receiving_tensor_view2 = local_tensor_dst[:, tensor_col_offset2:end_col_offset2]
    
    # Replacement: dist.P2POp -> tf.custom_gradient
    receive_op1 = tf.custom_gradient(lambda op, t, p, pg, tag: t)(None, receiving_tensor_view1, 0, None, tag1)
    receive_op2 = tf.custom_gradient(lambda op, t, p, pg, tag: t)(None, receiving_tensor_view2, 0, None, tag2)
    
    # Replacement: dist.batch_isend_irecv -> tf.custom_gradient
    batch_recv_op = tf.custom_gradient(lambda ops: [op() if callable(op) else op for op in ops])([receive_op1, receive_op2])

    # Assertions to verify shapes and basic properties
    assert dst_t1.shape == (batch_size, split_offsets[1] - split_offsets[0])
    assert dst_t2.shape == (batch_size, split_offsets[3] - split_offsets[2])
    assert receiving_tensor_view1.shape == dst_t1.shape
    assert receiving_tensor_view2.shape == dst_t2.shape
    
    # Verify the distance value was populated correctly
    assert torch.all(local_tensor_dst == dist_val)

if __name__ == "__main__":
    test_batch_isend_irecv_with_replacements()