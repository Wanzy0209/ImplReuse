import tensorflow as tf
import numpy as np

def test_diagonal_scalar_compilation():
    """
    Test case for tf.keras.ops.diagonal inspired by PyTorch Issue 164814.
    
    The original issue involves a divergence between eager and compiled modes
    when handling scalar tensors (0-dim), specifically related to stride/size
    dimensionality mismatches in as_strided operations.
    
    This test verifies that tf.keras.ops.diagonal, which can produce scalar
    outputs (e.g., from a 1x1 matrix) or empty outputs, behaves consistently
    in both eager and compiled (tf.function) modes.
    """
    
    # Case 1: Scalar result (1x1 matrix)
    # This mimics the 'var_node_0' scalar in the original bug that triggered
    # the stride dimensionality error.
    print("Testing 1x1 matrix (scalar result)...")
    input_scalar = tf.constant([[5.0]], dtype=tf.float32)

    def run_diagonal(x):
        # tf.keras.ops.diagonal on a 1x1 matrix returns a scalar (0-d tensor)
        val = tf.keras.ops.diagonal(x)
        # Ensure we perform an operation on the scalar to check validity
        result = val * 2.0
        return result

    # Eager execution
    result_eager = run_diagonal(input_scalar)
    print(f"  Eager result: {result_eager.numpy()}, Shape: {result_eager.shape}")

    # Compiled execution (tf.function)
    compiled_fn = tf.function(run_diagonal)
    result_compiled = compiled_fn(input_scalar)
    print(f"  Compiled result: {result_compiled.numpy()}, Shape: {result_compiled.shape}")

    # Assertions
    assert result_eager.shape == (), "Expected scalar output from 1x1 diagonal"
    assert tf.reduce_all(result_eager == result_compiled).numpy(), \
        "Divergence between eager and compiled modes for scalar diagonal"
    print("   Scalar diagonal test passed")

    # Case 2: Empty diagonal (offset out of bounds)
    # This relates to the _zeros branch in the similar API code snippet provided.
    print("Testing empty diagonal (offset out of bounds)...")
    input_matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)

    def run_diagonal_empty(x):
        # Offset 10 is out of bounds for a 2x2 matrix, should return empty tensor
        val = tf.keras.ops.diagonal(x, offset=10)
        return val

    result_eager = run_diagonal_empty(input_matrix)
    compiled_fn_empty = tf.function(run_diagonal_empty)
    result_compiled = compiled_fn_empty(input_matrix)

    assert result_eager.shape[0] == 0, "Expected empty diagonal"
    assert tf.reduce_all(result_eager == result_compiled).numpy(), \
        "Divergence between eager and compiled modes for empty diagonal"
    print("   Empty diagonal test passed")

if __name__ == "__main__":
    test_diagonal_scalar_compilation()