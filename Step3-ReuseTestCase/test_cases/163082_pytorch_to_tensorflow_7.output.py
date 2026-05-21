import torch
import tensorflow as tf
import numpy as np

def test_tf_keras_ops_diff():
    """
    Adapted test case based on PyTorch issue #163082.
    Original Issue: torch.nn.functional.normalize outputs vectors with norm>1 with torch.compile + cuda.
    Target API: tf.keras.ops.diff
    
    Note: While the original bug concerned normalization (norm > 1), this test verifies the 
    behavior of the similar API 'diff' using the same input data and compilation context 
    (tf.function) to check for numerical stability or correctness.
    """
    
    # Setup input matching the PyTorch bug report
    # Input: [[3.799999, 0.0, 0.0]]
    input_data = tf.constant([[3.799999, 0.0, 0.0]], dtype=tf.float32)
    
    print("Input vector:", input_data.numpy().tolist())

    # Define the function wrapped in tf.function (analogous to torch.compile)
    @tf.function
    def vec_diff_compiled(x):
        return tf.keras.ops.diff(x)

    # Define the function without compilation (eager execution)
    def vec_diff_eager(x):
        return tf.keras.ops.diff(x)

    # Run compiled version
    result_compiled = vec_diff_compiled(input_data)
    print("Diff result (tf.function/compiled):", result_compiled.numpy().tolist())

    # Run eager version
    result_eager = vec_diff_eager(input_data)
    print("Diff result (eager):", result_eager.numpy().tolist())

    # Expected result for diff on [3.799999, 0.0, 0.0] is [0.0 - 3.799999, 0.0 - 0.0] = [-3.799999, 0.0]
    expected_result = tf.constant([[-3.799999, 0.0]], dtype=tf.float32)

    # Verify that compiled and eager results match
    # Using a small epsilon for float comparison
    if not tf.reduce_all(tf.abs(result_compiled - result_eager) < 1e-6).numpy():
        raise AssertionError("Compiled and eager results differ significantly.")
    
    # Verify that the result matches the expected mathematical difference
    if not tf.reduce_all(tf.abs(result_compiled - expected_result) < 1e-6).numpy():
        raise AssertionError(f"Result {result_compiled.numpy()} does not match expected {expected_result.numpy()}.")

    print("Test passed: tf.keras.ops.diff behaves correctly.")

if __name__ == "__main__":
    test_tf_keras_ops_diff()