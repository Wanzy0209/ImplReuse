import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

def test_batch_isend_irecv_replacement():
    """
    Test case generated based on Issue ID: 161324.
    Replaces distributed communication APIs with TensorFlow custom_gradient
    and PyTorch math ops to preserve the structural logic of the bug report
    while adapting to the new APIs.
    """
    
    # Leverage the similar API: tf.compat.v1.executing_eagerly
    # to ensure the environment is correct for the replacement APIs.
    assert tf.compat.v1.executing_eagerly(), "Test requires TensorFlow eager execution"

    # Setup parameters mimicking the original bug report
    batch_size = 4
    total_columns = 10
    # Offsets for slicing views
    split_offsets = [0, 3, 7, 10] 

    # --- Rank 0 Logic (Sender) ---
    # Original: local_tensor = torch.randn(batch_size, total_columns)
    # Replacement: torch.nn.functional.max_pool2d
    # To use max_pool2d, we need a 4D input (N, C, H, W). We create a dummy input.
    # We choose dimensions such that the pooled output can be reshaped to 2D.
    dummy_input_rank0 = torch.randn(batch_size, 1, 5, 2)
    # Apply max_pool2d (kernel_size=2)
    local_tensor = F.max_pool2d(dummy_input_rank0, kernel_size=2)
    # Reshape to 2D to mimic the original tensor structure for slicing
    local_tensor = local_tensor.view(batch_size, -1)

    # Create views (slices) as in the original bug report
    dst_t1_pt = local_tensor[:, split_offsets[0]:split_offsets[1]]
    dst_t2_pt = local_tensor[:, split_offsets[2]:split_offsets[3]]

    # Convert PyTorch tensors to TensorFlow tensors for use with custom_gradient
    dst_t1 = tf.convert_to_tensor(dst_t1_pt.numpy())
    dst_t2 = tf.convert_to_tensor(dst_t2_pt.numpy())

    # Original: send_op1 = dist.P2POp(dist.isend, dst_t1, 1, process_group, tag1)
    # Replacement: tf.keras.ops.custom_gradient
    # We define the operation logic using the custom_gradient decorator.
    @tf.keras.ops.custom_gradient
    def send_op1(x):
        def grad(upstream):
            return upstream
        return x, grad

    @tf.keras.ops.custom_gradient
    def send_op2(x):
        def grad(upstream):
            return upstream
        return x, grad

    # Original: dist.batch_isend_irecv([send_op1, send_op2])
    # Replacement: tf.keras.ops.custom_gradient
    # We wrap the batch execution logic in a custom_gradient function.
    @tf.keras.ops.custom_gradient
    def batch_send_ops(t1, t2):
        # In the original, this initiates the send. Here we just pass through.
        def grad(upstream):
            return upstream, upstream
        return (t1, t2), grad

    # Execute the batch operation
    res1, res2 = batch_send_ops(dst_t1, dst_t2)
    print(f"Rank 0 processed tensors with shapes {res1.shape} and {res2.shape}")


    # --- Rank 1 Logic (Receiver) ---
    # Original: local_tensor_dst = torch.zeros(batch_size, total_columns)
    # Replacement: torch.slogdet
    # slogdet requires square matrices. We create a batch of identity matrices.
    # Shape: (batch_size, total_columns, total_columns)
    dummy_input_rank1 = torch.eye(total_columns).unsqueeze(0).repeat(batch_size, 1, 1)
    # slogdet returns (logabsdet, sign). We take logabsdet.
    # Result shape is (batch_size,).
    local_tensor_dst, _ = torch.slogdet(dummy_input_rank1)

    # Adapt slicing logic: original was 2D [:, start:end], result is 1D [start:end]
    receiving_tensor_view1_pt = local_tensor_dst[split_offsets[0]:split_offsets[1]]
    receiving_tensor_view2_pt = local_tensor_dst[split_offsets[2]:split_offsets[3]]

    # Convert to TF tensors
    receiving_tensor_view1 = tf.convert_to_tensor(receiving_tensor_view1_pt.numpy())
    receiving_tensor_view2 = tf.convert_to_tensor(receiving_tensor_view2_pt.numpy())

    # Original: receive_op1 = dist.P2POp(dist.irecv, receiving_tensor_view1, 0, process_group, tag1)
    # Replacement: tf.keras.ops.custom_gradient
    @tf.keras.ops.custom_gradient
    def receive_op1(x):
        def grad(upstream):
            return upstream
        return x, grad

    @tf.keras.ops.custom_gradient
    def receive_op2(x):
        def grad(upstream):
            return upstream
        return x, grad

    # Original: dist.batch_isend_irecv([receive_op1, receive_op2])
    # Replacement: tf.keras.ops.custom_gradient
    @tf.keras.ops.custom_gradient
    def batch_recv_ops(t1, t2):
        def grad(upstream):
            return upstream, upstream
        return (t1, t2), grad

    # Execute the batch operation
    res_recv1, res_recv2 = batch_recv_ops(receiving_tensor_view1, receiving_tensor_view2)
    print(f"Rank 1 processed tensors with shapes {res_recv1.shape} and {res_recv2.shape}")

    # Assertions to verify the test case runs and shapes are preserved
    assert res1.shape == dst_t1.shape
    assert res2.shape == dst_t2.shape
    assert res_recv1.shape == receiving_tensor_view1.shape
    assert res_recv2.shape == receiving_tensor_view2.shape

if __name__ == "__main__":
    test_batch_isend_irecv_replacement()