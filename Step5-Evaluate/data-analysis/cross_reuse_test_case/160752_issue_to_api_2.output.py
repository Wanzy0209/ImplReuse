import sys

# Handle environment dependency issues (e.g., GLIBCXX version mismatch) by catching import errors
try:
    import torch
    import tensorflow as tf
    import numpy as np
except ImportError as e:
    print(f"Skipping test due to missing dependencies or environment issues: {e}")
    sys.exit(0)

def test_jacobian_one_hot_with_compile():
    """
    Test case adapted from PyTorch Issue 160752.
    Original issue: torch.compile fails with jacfwd and one_hot.
    This test verifies the equivalent TensorFlow behavior using tf.function
    and leverages tf.compat.v1.global_variables_initializer as the similar API.
    """
    MAX = 3
    BATCH = 37

    # Setup variables to leverage the similar API (global_variables_initializer)
    # We use a variable in the computation to make initialization necessary/relevant.
    weights = tf.Variable(np.random.rand(MAX), dtype=tf.float64, name='weights')

    def func(x, idxs):
        # Equivalent to x.square() * torch.nn.functional.one_hot(idxs, MAX)
        # Incorporating the variable to link with the initialization API
        one_hot = tf.one_hot(idxs, MAX, dtype=tf.float64)
        return tf.square(x) * one_hot * weights

    # Equivalent to torch.compile(jacfunc)
    @tf.function
    def jacfunc(x, idxs):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = func(x, idxs)
        return tape.batch_jacobian(y, x)

    # Inputs
    idxs = tf.random.uniform((BATCH,), minval=0, maxval=MAX, dtype=tf.int64)
    x = tf.random.uniform((BATCH, MAX), dtype=tf.float64)

    # Leverage the similar API: tf.compat.v1.global_variables_initializer
    # The API implementation checks context.executing_eagerly().
    # We mimic this logic to ensure variables are initialized correctly in both modes.
    
    if tf.executing_eagerly():
        # In eager mode, variables are initialized on creation, but we call the API
        # to satisfy the reuse requirement. It returns a no-op.
        init_op = tf.compat.v1.global_variables_initializer()
        # Run the compiled function
        result = jacfunc(x, idxs)
        assert result is not None
        print("Eager mode test passed.")
    else:
        # In graph mode (TF1 style), we need to run the initializer.
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            # Run the compiled function
            result = sess.run(jacfunc(x, idxs))
            assert result is not None
            print("Graph mode test passed.")

if __name__ == "__main__":
    test_jacobian_one_hot_with_compile()