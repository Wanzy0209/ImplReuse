import torch
import tensorflow as tf
import tf.experimental.dtensor as dtensor

def test_dtensor_copy_to_mesh_with_diagonal_fill():
    """
    Adapted test case for tf.experimental.dtensor.copy_to_mesh.
    Original PyTorch bug: Inductor fails on t2.fill_diagonal_(t1.item()) inside torch.compile.
    This test verifies that the TensorFlow equivalent operations work correctly
    when distributing the result tensor using copy_to_mesh.
    """
    
    # 1. Setup Mesh (using CPU for portability)
    # In a real distributed scenario, this would span multiple devices.
    devices = tf.config.list_physical_devices('CPU')
    if not devices:
        print("No CPU devices found, test cannot run.")
        return

    mesh = dtensor.create_mesh([("batch", 1)], devices=devices)
    layout = dtensor.Layout.replicated(mesh, rank=2)

    # 2. Define the function (adapted from PyTorch foo)
    def foo(arg0, arg1):
        # t0 = arg0
        # t2 = t0.clone()
        t2 = tf.identity(arg0)

        # t2.fill_diagonal_(t1.item())
        # TensorFlow equivalent: tf.linalg.set_diag
        # arg1 is a 0-d tensor (scalar). set_diag expects a rank-1 tensor for the diagonal 
        # if the input is rank-2. We reshape arg1 to [1] to match the diagonal size of (1,1).
        diagonal_values = tf.reshape(arg1, [1])
        t2 = tf.linalg.set_diag(t2, diagonal_values)

        return t2

    # 3. Setup Inputs
    # PyTorch: arg0 = torch.empty([1, 1], ...), arg1 = torch.empty([], ...)
    # We use zeros/constant to ensure deterministic behavior for the test.
    arg0 = tf.zeros([1, 1], dtype=tf.float32)
    arg1 = tf.constant(5.0, dtype=tf.float32) # 0-d tensor

    # 4. Run Eager
    out_eager = foo(arg0, arg1)
    print("Eager Result:", out_eager.numpy())
    
    # Verify eager result
    expected_eager = tf.constant([[5.0]], dtype=tf.float32)
    assert tf.reduce_all(out_eager == expected_eager).numpy(), "Eager execution failed!"
    print('Eager Success! ')

    # 5. Run with Similar API (copy_to_mesh)
    # The original bug involved torch.compile. Here we test the distribution API
    # handling the tensor resulting from the diagonal fill operation.
    try:
        # We wrap the operation in a GradientTape to test backward pass compatibility
        # similar to the original .backward() call.
        with tf.GradientTape() as tape:
            # Variables are needed for gradient tracking
            var0 = tf.Variable(arg0)
            var1 = tf.Variable(arg1)
            
            # Execute logic
            res = foo(var0, var1)
            
            # Apply the target API: copy_to_mesh
            out_distributed = dtensor.copy_to_mesh(res, layout)
            
            # Calculate loss for backward pass
            loss = tf.reduce_sum(out_distributed)

        # 6. Backward Pass
        grads = tape.gradient(loss, [var0, var1])
        print("Gradients:", grads)
        
        # Verify distributed result
        # Note: Depending on the layout, we might need to check local components,
        # but for replicated layout on a single device, it should match.
        assert tf.reduce_all(out_distributed == expected_eager).numpy(), "Distributed result mismatch!"
        
        print('Copy to Mesh & Backward Success! ')

    except Exception as e:
        print(f"Test Failed with error: {e}")
        raise

if __name__ == '__main__':
    test_dtensor_copy_to_mesh_with_diagonal_fill()