import torch
import tensorflow as tf

def test_cuda_indexing_replacement():
    # Check for CUDA availability to respect the original bug's context
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # 1. Replace torch.randint with torch.fmin
    # Original: mask = torch.randint(0, 20, (4, 87, 1056, 736), device="cuda")
    # We create two tensors and compute the element-wise minimum to generate the data.
    t1 = torch.ones((4, 87, 1056, 736), device="cuda") * 10
    t2 = torch.ones((4, 87, 1056, 736), device="cuda") * 20
    mask = torch.fmin(t1, t2)

    # 2. Replace torch.tensor with tf.compat.v1.linalg.diag
    # Original: to_apply = torch.tensor([True, False, False, True], device="cuda")
    # We create a diagonal matrix representing the selection mask.
    # 1.0 represents True, 0.0 represents False.
    to_apply = tf.compat.v1.linalg.diag([1.0, 0.0, 0.0, 1.0])

    # 3. Replace Tensor indexing (mask[to_apply]) with tf.keras.ops.custom_gradient
    # We need to convert the PyTorch tensor to a TensorFlow tensor to operate within the TF context.
    mask_tf = tf.convert_to_tensor(mask.cpu().numpy())

    @tf.keras.ops.custom_gradient
    def custom_indexing_op(data, diag_mask):
        # data: The large tensor (4, 87, 1056, 736)
        # diag_mask: The diagonal matrix (4, 4)

        # Extract the boolean mask from the diagonal
        bool_mask = tf.linalg.diag_part(diag_mask) > 0  # Shape (4,)

        # Perform the indexing logic (Gather rows where mask is True)
        # This mimics mask[to_apply] which selects rows 0 and 3.
        indices = tf.where(bool_mask)[:, 0]
        output = tf.gather(data, indices)

        # Define the gradient function for the custom operation
        def grad(upstream):
            # Scatter the upstream gradients back to the original indices
            grad_data = tf.zeros_like(data)
            grad_data = tf.tensor_scatter_nd_add(grad_data, tf.expand_dims(indices, 1), upstream)
            return grad_data, None

        return output, grad

    # Execute the operation
    result = custom_indexing_op(mask_tf, to_apply)

    # Assertions to verify the logic
    # The result should have 2 rows (corresponding to the 2 True values in the mask)
    assert result.shape == (2, 87, 1056, 736)
    
    # Verify the content (values should be 10.0 from the fmin operation)
    assert tf.reduce_all(result == 10.0)

if __name__ == "__main__":
    test_cuda_indexing_replacement()