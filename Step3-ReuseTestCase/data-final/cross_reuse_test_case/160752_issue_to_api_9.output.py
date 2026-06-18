import torch
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

def test_jacobian_one_hot_compile():
    """
    Test case adapted from PyTorch Issue 160752.
    Verifies that computing the Jacobian of a function involving one_hot
    works under compilation (tf.function), leveraging cast_to_floatx for type safety.
    """
    MAX = 3
    BATCH = 37

    def func(x, idxs):
        # Leverage the similar API to ensure x is the default float type
        x = K.cast_to_floatx(x)
        return tf.square(x) * tf.one_hot(idxs, MAX, dtype=x.dtype)

    def jacfunc(x, idxs):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = func(x, idxs)
        return tape.jacobian(y, x)

    # Compile the function (equivalent to torch.compile)
    # Using jit_compile=True to stress-test the compilation path
    jacfunc_compiled = tf.function(jacfunc, jit_compile=True)

    # Prepare inputs
    # Using float64 to test the casting logic provided by the similar API
    idxs = tf.constant(np.random.randint(0, MAX, size=(BATCH,)), dtype=tf.int64)
    x = tf.constant(np.random.rand(BATCH, MAX), dtype=tf.float64)

    # Run the compiled function
    try:
        result = jacfunc_compiled(x, idxs)
        # Verify output shape
        # Jacobian of (BATCH, MAX) w.r.t (BATCH, MAX) is (BATCH, MAX, BATCH, MAX)
        assert result.shape == (BATCH, MAX, BATCH, MAX)
        print("Test Passed.")
    except Exception as e:
        print(f"Test Failed with error: {e}")
        raise

if __name__ == "__main__":
    test_jacobian_one_hot_compile()