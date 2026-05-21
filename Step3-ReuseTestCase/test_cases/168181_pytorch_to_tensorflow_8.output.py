import unittest
import tensorflow as tf
import numpy as np

class TestTfKerasNameScope(unittest.TestCase):
    def test_name_scope_gpu_to_cpu(self):
        """
        Adapted from PyTorch test_triton_kernel_to_cpu.
        Verifies that operations wrapped in tf.keras.name_scope
        correctly handle GPU computation followed by a CPU transfer and arithmetic.
        """
        # Setup inputs (GPU by default if available)
        x = tf.random.normal((4, 4))
        y = tf.random.normal((4, 4))

        # Original logic: out = kernel(x, y); out_cpu = out.cpu() + 1
        # We adapt this to use the target API: tf.keras.name_scope
        with tf.keras.name_scope("custom_kernel_simulation"):
            # Simulate the Triton kernel (addition)
            # In the original bug, 'out' was modified in-place by the kernel.
            # Here we perform the addition operation.
            out = x + y
            
            # Simulate .cpu() + 1
            # Explicitly place the operation on CPU to mimic the device transfer
            with tf.device("/CPU:0"):
                result = out + 1

        # Verify correctness
        # The original test compared eager vs compiled outputs.
        # Here we verify the result against the expected mathematical computation.
        expected = (x.numpy() + y.numpy()) + 1
        np.testing.assert_allclose(result.numpy(), expected)

if __name__ == '__main__':
    unittest.main()