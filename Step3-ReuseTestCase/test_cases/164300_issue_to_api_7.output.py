import torch
import tensorflow as tf
import functools

# Test case to verify if functools.partial works with tf.keras.backend
# inside a compiled context (tf.function), mirroring the PyTorch issue
# where partial'ed context_fn failed in torch.compile.

def test_keras_backend_with_functools_partial():
    # Define a custom operation using tf.keras.backend
    def custom_op(x, multiplier):
        return tf.keras.backend.square(x) * multiplier

    # Create a functools.partial of the operation
    # This mimics the 'context_fn1 = functools.partial(...)' in the bug report
    partial_op = functools.partial(custom_op, multiplier=2.0)

    # Define a function decorated with tf.function (equivalent to torch.compile)
    # that uses the partial'd function.
    # tf.keras.backend.in_train_phase is a function that accepts callables,
    # making it a good candidate to test passing the partial object.
    @tf.function
    def compiled_graph(x, training):
        return tf.keras.backend.in_train_phase(
            partial_op,
            functools.partial(custom_op, multiplier=1.0),
            training=training
        )

    # Input tensor
    x = tf.constant([1.0, 2.0, 3.0])

    # Test in training phase (uses multiplier=2.0)
    # Expected: [1^2*2, 2^2*2, 3^2*2] = [2, 8, 18]
    result_train = compiled_graph(x, training=True)
    expected_train = tf.constant([2.0, 8.0, 18.0])
    
    # Test in inference phase (uses multiplier=1.0)
    # Expected: [1^2*1, 2^2*1, 3^2*1] = [1, 4, 9]
    result_test = compiled_graph(x, training=False)
    expected_test = tf.constant([1.0, 4.0, 9.0])

    # Assertions
    # Using tf.reduce_all to check tensor equality
    assert tf.reduce_all(tf.equal(result_train, expected_train)).numpy(), "Training phase result mismatch"
    assert tf.reduce_all(tf.equal(result_test, expected_test)).numpy(), "Inference phase result mismatch"

    print("Test passed: functools.partial works with tf.keras.backend inside tf.function.")

if __name__ == "__main__":
    test_keras_backend_with_functools_partial()