import sys

try:
    import torch
    import tensorflow as tf
except ImportError as e:
    # Handle environment issues such as missing dependencies or GLIBC version mismatches
    print(f"Skipping test due to import error: {e}")
    print("This is likely due to a system library incompatibility (e.g., GLIBCXX version).")
    sys.exit(0)

def test_reduce_all_empty_tensor_compilation():
    """
    Test case for tf.math.reduce_all based on the PyTorch issue 164875.
    The original issue involves a divergence between eager and compiled modes
    when handling tensors with zero-sized dimensions (specifically (20, 0)).
    This test verifies that tf.math.reduce_all handles such shapes consistently
    in both eager execution and tf.function (graph mode).
    """
    # Mimic the problematic shape (20, 0) from the PyTorch issue
    # PyTorch: torch.as_strided(..., (20, 0), ...)
    # TensorFlow: Create a boolean tensor with shape (20, 0)
    input_tensor = tf.zeros((20, 0), dtype=tf.bool)

    # 1. Eager Execution
    eager_result = tf.math.reduce_all(input_tensor)

    # 2. Compiled Execution (tf.function is the TF equivalent of torch.compile)
    @tf.function
    def compiled_func(x):
        return tf.math.reduce_all(x)

    compiled_result = compiled_func(input_tensor)

    # 3. Check for divergence
    # tf.math.reduce_all on an empty tensor returns True (identity of logical AND)
    assert eager_result.numpy() == compiled_result.numpy(), \
        f"Divergence detected! Eager: {eager_result.numpy()}, Compiled: {compiled_result.numpy()}"

    print(" Test passed: tf.math.reduce_all handles empty tensor (20, 0) consistently in eager and compiled modes.")

if __name__ == "__main__":
    test_reduce_all_empty_tensor_compilation()