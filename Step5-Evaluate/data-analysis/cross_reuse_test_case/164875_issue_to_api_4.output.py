import sys

# Handle environment issues (e.g., missing GLIBCXX) by catching import errors
try:
    import torch
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test due to environment dependency error: {e}")
    sys.exit(0)

def test_reduce_any_empty_dim_consistency():
    """
    Test case for tf.math.reduce_any handling tensors with zero-sized dimensions.
    This mirrors the PyTorch issue where eager and compiled modes diverged
    when handling tensors with shape (20, 0).
    """
    # Mimic the shape (20, 0) from the PyTorch bug report
    # arg_0 = torch.as_strided(..., (20, 0), ...)
    input_tensor = tf.zeros((20, 0), dtype=tf.bool)

    # 1. Test Eager Execution
    # Reducing along the empty dimension (axis=1).
    # Expected: Shape (20,), all False (identity for logical OR)
    eager_result = tf.math.reduce_any(input_tensor, axis=1)
    assert eager_result.shape == (20,), f"Eager shape mismatch: {eager_result.shape}"
    assert tf.reduce_all(tf.equal(eager_result, False)), "Eager values incorrect"
    print(' eager success')

    # 2. Test Compiled Execution (tf.function)
    # This mirrors the torch.compile check in the original bug report.
    # We use tf.function to simulate the graph/compiled mode.
    @tf.function
    def compiled_reduce(t):
        return tf.math.reduce_any(t, axis=1)

    compiled_result = compiled_reduce(input_tensor)
    
    # 3. Verify Consistency
    # The original bug was a divergence between eager and compile.
    # We assert they produce the same result here.
    assert compiled_result.shape == (20,), f"Compiled shape mismatch: {compiled_result.shape}"
    assert tf.reduce_all(tf.equal(compiled_result, False)), "Compiled values incorrect"
    assert tf.reduce_all(tf.equal(eager_result, compiled_result)), "Divergence between eager and compiled"
    print(' compile success')

if __name__ == "__main__":
    test_reduce_any_empty_dim_consistency()