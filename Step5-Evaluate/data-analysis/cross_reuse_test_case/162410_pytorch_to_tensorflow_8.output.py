import sys
import numpy as np

# Handle environment/dependency issues gracefully
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: Failed to import TensorFlow due to environment incompatibility.")
    print(f"Error details: {e}")
    print("This is likely caused by a missing system library (e.g., GLIBCXX_3.4.29 not found).")
    sys.exit(0)

import torch

def test_inplace_mutation_with_name_scope():
    """
    Adapted test case from PyTorch issue #162410.
    Verifies that in-place mutations (assign) combined with view operations (reverse)
    behave correctly when executed inside a tf.keras.name_scope under graph compilation
    (tf.function), similar to the torch.compile scenario.
    """
    
    # Define the eager version (Reference)
    def f_eager(x, y):
        with tf.keras.name_scope("mutation_scope"):
            # PyTorch: x.copy_(x.flip(1))
            # TensorFlow: x.assign(tf.reverse(x, axis=[1]))
            # Note: x must be a tf.Variable for in-place assignment
            x.assign(tf.reverse(x, axis=[1]))
        
        with tf.keras.name_scope("computation_scope"):
            # PyTorch: y = y.sum(dim=1, keepdim=True) + y
            y = tf.reduce_sum(y, axis=1, keepdims=True) + y
            
        return x + y

    # Define the compiled version using tf.function (Similar to torch.compile)
    # We wrap the logic in tf.keras.name_scope as requested by the API mapping
    @tf.function
    def f_compiled(x, y):
        with tf.keras.name_scope("mutation_scope"):
            x.assign(tf.reverse(x, axis=[1]))
            
        with tf.keras.name_scope("computation_scope"):
            y = tf.reduce_sum(y, axis=1, keepdims=True) + y
            
        return x + y

    # Setup inputs
    # PyTorch: torch.randn(20, 1024 * 1024, device="cuda")
    shape = (20, 1024 * 1024)
    initial_x_val = np.random.randn(*shape).astype(np.float32)
    initial_y_val = np.random.randn(*shape).astype(np.float32)

    # Run Eager (Reference)
    x_var_eager = tf.Variable(initial_x_val)
    y_tensor_eager = tf.constant(initial_y_val)
    ref = f_eager(x_var_eager, y_tensor_eager)

    # Run Compiled (Actual)
    # We must use fresh variables/tensors because the operation is in-place
    x_var_compiled = tf.Variable(initial_x_val)
    y_tensor_compiled = tf.constant(initial_y_val)
    act = f_compiled(x_var_compiled, y_tensor_compiled)

    # Assert close to check for numerical issues/incorrect fusion
    # This mimics torch.testing.assert_close(ref, act)
    np.testing.assert_allclose(ref.numpy(), act.numpy(), rtol=1e-5, atol=1e-5)
    print("Test passed: Eager and Compiled results match within tf.keras.name_scope.")

if __name__ == "__main__":
    test_inplace_mutation_with_name_scope()