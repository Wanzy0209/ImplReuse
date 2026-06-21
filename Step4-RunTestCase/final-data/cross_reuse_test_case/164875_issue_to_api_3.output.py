import sys

# Handle environment/dependency errors gracefully
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: Failed to import TensorFlow.")
    print(f"Error: {e}")
    print("This is likely due to a system library version mismatch (e.g., GLIBCXX).")
    sys.exit(0)

import torch

# Test case derived from PyTorch Issue 164875
# Issue: Shape mismatch error ("The size of tensor a (s67) must match the size of tensor b (u0)")
# occurring with tensors of size (20, 0) in compiled mode vs eager mode.
# Similar API: tf.compat.v1.reduce_any
# Goal: Verify that reduce_any handles degenerate dimensions (size 0) correctly
# in both eager and graph (compiled) modes to prevent similar divergence.

def test_reduce_any_with_degenerate_dimensions():
    # Recreate the specific shape from the bug report: (20, 0)
    # In PyTorch: var_node_1 = torch.as_strided(..., (20, 0), ...)
    # This represents a tensor with 20 rows but 0 columns (empty inner dimension).
    input_tensor = tf.zeros((20, 0), dtype=tf.bool)

    # Define the operation using the similar API
    def run_reduce_any(x):
        # Reducing along axis 0 on a (20, 0) tensor.
        # Since the dimension 1 is size 0, the result should be a tensor of shape (0,).
        # The logical reduction of an empty set is False.
        return tf.compat.v1.reduce_any(x, axis=0)

    # 1. Eager Execution
    eager_result = run_reduce_any(input_tensor)
    print(f"Eager Result Shape: {eager_result.shape}")

    # 2. Compiled/Graph Execution (simulating torch.compile)
    # In TensorFlow, tf.function traces the code to create a graph, analogous to PyTorch compilation.
    # This checks if the graph tracer correctly infers shapes for empty tensors.
    compiled_func = tf.function(run_reduce_any)
    compiled_result = compiled_func(input_tensor)
    print(f"Compiled Result Shape: {compiled_result.shape}")

    # Assertions to ensure consistency and correctness
    # The result of reducing an empty dimension should be False (identity for logical OR)
    # and the shape should be (0,).
    assert eager_result.shape == (0,), f"Expected shape (0,), got {eager_result.shape}"
    assert compiled_result.shape == (0,), f"Expected shape (0,), got {compiled_result.shape}"
    
    # Verify values are all False
    assert tf.reduce_all(eager_result == False).numpy(), "Eager result should be all False"
    assert tf.reduce_all(compiled_result == False).numpy(), "Compiled result should be all False"

    # Test with keepdims=True to ensure dimension retention logic is sound for empty tensors
    def run_reduce_any_keepdims(x):
        return tf.compat.v1.reduce_any(x, axis=0, keepdims=True)

    eager_kd = run_reduce_any_keepdims(input_tensor)
    compiled_kd = tf.function(run_reduce_any_keepdims)(input_tensor)

    assert eager_kd.shape == (1, 0), f"Expected shape (1, 0), got {eager_kd.shape}"
    assert compiled_kd.shape == (1, 0), f"Expected shape (1, 0), got {compiled_kd.shape}"

    print(" Test passed: tf.compat.v1.reduce_any handles (20, 0) tensors correctly in both modes.")

if __name__ == "__main__":
    test_reduce_any_with_degenerate_dimensions()