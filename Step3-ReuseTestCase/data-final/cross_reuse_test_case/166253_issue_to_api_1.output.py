import torch
import tensorflow as tf

class TestTensorFillWithJIT(tf.test.TestCase):
    def test_tf_fill_jit_float64(self):
        """
        Test case adapted from PyTorch issue #166253.
        Verifies that tf.fill (similar to torch.full) correctly updates values
        when called with different inputs inside a JIT-compiled function (XLA).
        """
        # Leverage the similar API to check if the environment supports XLA/JIT
        if not tf.test.is_built_with_xla():
            self.skipTest("Test is only applicable with XLA")

        # Original bug reproduction logic adapted for TensorFlow
        # torch.full((2,), x, dtype=torch.float64) -> tf.fill((2,), x)
        def func_nojit(x):
            return tf.fill((2,), x)

        # torch.compile(func_nojit) -> @tf.function(jit_compile=True)
        @tf.function(jit_compile=True)
        def func_jit(x):
            return tf.fill((2,), x)

        # Define inputs matching the original issue (float64)
        x1 = tf.constant(5.0, dtype=tf.float64)
        x2 = tf.constant(10.0, dtype=tf.float64)

        # Expected behavior: values should match the input argument
        # Test non-jit version
        self.assertAllEqual(func_nojit(x1), [5.0, 5.0])
        self.assertAllEqual(func_nojit(x2), [10.0, 10.0])

        # Test jit version (reproducing the potential bug scenario)
        # In the PyTorch bug, the second call returned [5., 5.] instead of [10., 10.]
        self.assertAllEqual(func_jit(x1), [5.0, 5.0])
        self.assertAllEqual(func_jit(x2), [10.0, 10.0])

if __name__ == '__main__':
    tf.test.main()