import torch
import tensorflow as tf
import numpy as np

# Define the computation function similar to the PyTorch example.
# Note: In TensorFlow, in-place mutation requires a tf.Variable.
def computation(x, y):
    # x.copy_(x.flip(1)) -> x.assign(tf.reverse(x, axis=[1]))
    # This mimics the in-place mutation that caused the fusion issue in PyTorch.
    x.assign(tf.reverse(x, axis=[1]))
    
    # y.sum(dim=1, keepdim=True) + y
    y = tf.reduce_sum(y, axis=1, keepdims=True) + y
    
    return x + y

def main():
    # Initialize TPU system if available.
    # This is required for tf.compat.v1.tpu.rewrite to function.
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("TPU initialized.")
    except ValueError:
        print("No TPU found. This test requires a TPU environment to run tf.compat.v1.tpu.rewrite.")
        return

    # Generate random data matching the PyTorch test case dimensions
    # PyTorch: torch.randn(20, 1024 * 1024)
    np.random.seed(42)
    x_np = np.random.randn(20, 1024 * 1024).astype(np.float32)
    y_np = np.random.randn(20, 1024 * 1024).astype(np.float32)

    # 1. Eager Execution (Reference)
    # x needs to be a Variable to support the in-place assign operation
    x_var_eager = tf.Variable(x_np)
    y_tensor_eager = tf.constant(y_np)
    
    ref = computation(x_var_eager, y_tensor_eager)

    # 2. Compiled Execution using tf.compat.v1.tpu.rewrite
    # Reset inputs for the compiled run
    x_var_comp = tf.Variable(x_np)
    y_tensor_comp = tf.constant(y_np)

    # Call the target API
    # The API expects a computation and a list of inputs.
    # According to docs, it returns a list of tensors.
    compiled_result = tf.compat.v1.tpu.rewrite(
        computation, 
        inputs=[x_var_comp, y_tensor_comp]
    )
    
    # Extract the tensor from the list returned by rewrite
    act = compiled_result[0]

    # Verify results
    # torch.testing.assert_close -> tf.debugging.assert_near
    try:
        tf.debugging.assert_near(ref, act, message="Eager and TPU compiled results differ")
        print("Test passed: Eager and TPU compiled results are close.")
    except tf.errors.InvalidArgumentError as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()