import torch
import tensorflow as tf
import tensorflow.experimental.dtensor as dtensor

def test_dtensor_copy_to_mesh_complex_layout():
    """
    Adapted from PyTorch Issue 166319.
    Verifies that tf.experimental.dtensor.copy_to_mesh handles tensors
    resulting from complex operations (squeeze, chunk, gather, broadcasting)
    without crashing, similar to the eager/compile divergence scenario in PyTorch.
    """
    
    # 1. Setup Mesh
    # We use a single CPU mesh to ensure the test runs in most environments.
    # In a distributed setting, this would span multiple devices.
    try:
        mesh = dtensor.create_mesh([("x", 1)], devices=["CPU:0"])
    except Exception as e:
        print(f"Skipping test: Could not create DTensor mesh. {e}")
        return

    # 2. Replicate the tensor construction logic from the PyTorch bug report
    
    # arg_0: size=(9, 1, 15, 4), stride=(60, 60, 0, 1)
    # The stride 0 on dim 2 implies broadcasting. 
    # We simulate this by creating a smaller tensor and broadcasting it.
    base_arg_0 = tf.random.uniform((9, 1, 1, 4), minval=5, maxval=30, dtype=tf.int32)
    arg_0 = tf.broadcast_to(base_arg_0, (9, 1, 15, 4))

    # var_node_3 = torch.squeeze(var_node_4) -> size=(9, 15, 4)
    var_node_3 = tf.squeeze(arg_0, axis=1)

    # var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0] -> size=(9, 15, 1)
    # PyTorch chunk splits along a dimension. TF split is equivalent.
    split_nodes = tf.split(var_node_3, 4, axis=2)
    var_node_2 = split_nodes[0]

    # var_node_1 = torch.squeeze(var_node_2) -> size=(9, 15)
    var_node_1 = tf.squeeze(var_node_2, axis=2)

    # arg_1: size=(20, 15), stride=(15, 1), dtype=int64
    arg_1 = tf.random.uniform((20, 15), minval=5, maxval=30, dtype=tf.int64)

    # arg_2: size=(18, 15), stride=(15, 1), dtype=int64
    # (Present in original args, used for size reference in randint)
    arg_2 = tf.random.uniform((18, 15), minval=5, maxval=30, dtype=tf.int64)

    # _input_size_var_node_6 = var_node_7.size(0) -> 20
    input_size = tf.shape(arg_1)[0]

    # _index_var_node_6 = torch.randint(0, _input_size_var_node_6, (18, 15))
    # Generate random indices for gather
    _index_var_node_6 = tf.random.uniform((18, 15), minval=0, maxval=tf.cast(input_size, tf.int32), dtype=tf.int32)

    # var_node_6 = torch.gather(var_node_7, 0, _index_var_node_6)
    # PyTorch gather with dim=0: out[i][j] = input[index[i][j]][j]
    # TF gather with batch_dims=1: out[i][j] = params[indices[i][j], j]
    var_node_6 = tf.gather(arg_1, _index_var_node_6, batch_dims=1, axis=0)

    # var_node_5 = torch.chunk(var_node_6, 2, dim=0)[0] -> size=(9, 15)
    split_nodes_5 = tf.split(var_node_6, 2, axis=0)
    var_node_5 = split_nodes_5[0]

    # var_node_0 = torch.mul(var_node_1, var_node_5)
    # Cast int32 to int64 to match PyTorch behavior (int64 * int64 -> int64)
    var_node_1_cast = tf.cast(var_node_1, tf.int64)
    var_node_0 = tf.multiply(var_node_1_cast, var_node_5)

    # Sentinel logic to ensure gradient computation is tracked
    # In PyTorch: sentinel = torch.tensor(1.0, requires_grad=True)
    # In TF: We use tf.GradientTape
    sentinel = tf.Variable(1.0, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        # Cast result to float for multiplication with sentinel
        result_float = tf.cast(var_node_0, tf.float32)
        result = result_float * sentinel
        
        # PyTorch check: if result.is_complex(): result = result.real
        # TF tensors have fixed dtypes, so we check the dtype
        if result.dtype.is_complex:
            result = tf.math.real(result)

    # Calculate gradients to ensure the graph is valid
    grads = tape.gradient(result, sentinel)
    assert grads is not None

    # 3. Test the Target API: tf.experimental.dtensor.copy_to_mesh
    
    # Define a layout. We use a fully replicated layout for simplicity.
    layout = dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh)
    
    try:
        # The API expects a regular Tensor and converts it to a DTensor
        dtensor_result = dtensor.copy_to_mesh(result, layout)
        
        # Verify the conversion succeeded and values are preserved
        # Convert back to regular tensor to check values
        final_tensor = dtensor_result.to_tensor()
        
        # Assert values match
        # Note: DTensor might change sharding, but values must be identical
        assert tf.reduce_all(result == final_tensor).numpy()
        
        print(" copy_to_mesh success: Complex tensor handled correctly.")
        print(f"   Input shape: {result.shape}, Output layout: {layout}")
        
    except Exception as e:
        print(f" copy_to_mesh failed: {e}")
        raise

if __name__ == "__main__":
    test_dtensor_copy_to_mesh_complex_layout()