import torch
import tensorflow as tf
from tensorflow.keras.ops import add

# The original issue involves torch.cond causing a segmentation fault under torch.compile.
# This test case adapts the reproduction logic to the TensorFlow ecosystem.
# It verifies the similar API (tf.keras.ops.add) within a conditional execution
# context (tf.cond) under a compiled context (tf.function), mirroring the
# structure of the original bug report.

@tf.function(jit_compile=True)
def compiled_conditional_add(pred, a, b):
    """
    Replicates the structure of the failing PyTorch code:
    1. Compiled function (torch.compile -> @tf.function)
    2. Conditional logic (torch.cond -> tf.cond)
    3. Operation execution (Tensor ops -> tf.keras.ops.add)
    """
    # Branch 1: Perform addition using the similar API
    def true_fn():
        return add(a, b)

    # Branch 2: Perform addition with zero (mimicking the 'else' logic in the bug report)
    def false_fn():
        return add(a, tf.zeros_like(b))

    # Execute based on predicate
    return tf.cond(pred, true_fn, false_fn)

def test_tf_keras_ops_add_in_cond():
    # Inputs
    x = tf.constant([1.0, 2.0, 3.0])
    y = tf.constant([4.0, 5.0, 6.0])

    # Test True Branch
    result_true = compiled_conditional_add(tf.constant(True), x, y)
    expected_true = tf.constant([5.0, 7.0, 9.0])
    assert tf.reduce_all(tf.equal(result_true, expected_true)).numpy(), "True branch assertion failed"

    # Test False Branch
    result_false = compiled_conditional_add(tf.constant(False), x, y)
    expected_false = tf.constant([1.0, 2.0, 3.0])
    assert tf.reduce_all(tf.equal(result_false, expected_false)).numpy(), "False branch assertion failed"

    print("Test passed: tf.keras.ops.add behaves correctly inside tf.cond with compilation.")

if __name__ == "__main__":
    test_tf_keras_ops_add_in_cond()