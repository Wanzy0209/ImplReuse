import torch
import tensorflow as tf

# The original bug report involved a tensor with shape (20, 0) causing a size mismatch 
# during eager/compile divergence in PyTorch's torch.add.
# Here we adapt the test to verify the behavior of the similar TensorFlow API 
# tf.compat.v1.no_regularizer with the same tensor characteristics.

# Note: tf.compat.v1.no_regularizer is a utility function that returns None to prevent 
# regularization. It does not perform tensor arithmetic like torch.add, so the specific 
# size mismatch error cannot be reproduced identically. However, we verify that the API 
# handles the specific tensor shape gracefully in both eager and graph modes.

def test_no_regularizer_with_zero_dim():
    # Mimic the tensor shape (20, 0) from the original PyTorch bug report
    # PyTorch: arg_0 = torch.as_strided(..., (20, 0), ...)
    # TensorFlow equivalent:
    tensor_input = tf.zeros((20, 0), dtype=tf.int64)

    # 1. Test Eager Execution
    # Corresponds to result_original = fuzzed_program(*args)
    result_eager = tf.compat.v1.no_regularizer(tensor_input)
    print(f"Eager result: {result_eager}")
    assert result_eager is None, "tf.compat.v1.no_regularizer should return None in eager mode"

    # 2. Test Compiled Execution (tf.function)
    # Corresponds to compiled_program = torch.compile(...)
    @tf.function
    def compiled_no_regularizer(t):
        return tf.compat.v1.no_regularizer(t)

    result_compiled = compiled_no_regularizer(tensor_input)
    print(f"Compiled result: {result_compiled}")
    assert result_compiled is None, "tf.compat.v1.no_regularizer should return None in compiled mode"

    print(" Test passed: API handles the specific tensor shape correctly in both modes.")

if __name__ == "__main__":
    test_no_regularizer_with_zero_dim()