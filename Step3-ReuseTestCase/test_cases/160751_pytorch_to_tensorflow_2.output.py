import torch
import tensorflow as tf
import tf.experimental.dtensor as dtensor

def func():
    # Setup a simple mesh for the DTensor operation
    # Using CPU to ensure the test runs in standard environments
    mesh = dtensor.create_mesh([("x", 1)], devices=["CPU:0"])
    
    # Create a tensor with a negative value to trigger the assertion failure
    a = tf.constant([1.0, -2.0])
    
    # In TensorFlow graph mode (tf.function), we use tf.Assert to check conditions.
    # This mimics the 'assert result' logic in the PyTorch code.
    # The PyTorch bug was that torch.cuda.synchronize() was removed, causing 
    # missed exceptions. Here we verify that the assertion (and synchronization 
    # implied by control_dependencies) is respected when using copy_to_mesh.
    check = tf.debugging.assert_positive(a, message="should throw")
    
    with tf.control_dependencies([check]):
        layout = dtensor.Layout([dtensor.UNSHARDED], mesh)
        # Call the similar API: copy_to_mesh
        # This operation should respect the control dependency and not execute
        # if the assertion fails.
        result = dtensor.copy_to_mesh(a, layout=layout)
    
    # This print statement should not be reached if the assertion works correctly
    tf.print("should not run")

def test_fn():
    # Wrap the function in tf.function to enable graph compilation/optimization,
    # similar to torch.compile with backend="aot_eager".
    f_c = tf.function(func)
    
    try:
        f_c()
        # If we reach here, the exception was missed (reproducing the bug behavior)
        print("BUG: Exception was missed, code continued execution.")
    except Exception as e:
        # Expected behavior: The assertion error is caught
        print(f"SUCCESS: Caught expected exception: {e}")

if __name__ == "__main__":
    test_fn()