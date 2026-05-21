import torch
import unittest
import tensorflow as tf
import numpy as np

class TestNameScope(unittest.TestCase):
    @unittest.skipIf(not tf.test.is_built_with_cuda(), "Requires GPU")
    def test_name_scope_gpu_to_cpu(self):
        """
        Adapted from PyTorch test_triton_kernel_to_cpu.
        Verifies that operations wrapped in tf.name_scope handle
        device transfers (GPU -> CPU) correctly in both eager and graph modes.
        """
        def f(x, y):
            # Using tf.name_scope to group the custom kernel operation
            with tf.name_scope("triton_kernel_sim"):
                # Simulating the user-defined kernel output
                out = tf.add(x, y, name="add_kernel")
            
            # Simulating .cpu() + 1
            # In TensorFlow, we explicitly place the operation on CPU
            with tf.device("/CPU:0"):
                out_cpu = out + 1
            return out_cpu

        # Create inputs on GPU
        with tf.device("/GPU:0"):
            x = tf.random.normal((4, 4), seed=42)
            y = tf.random.normal((4, 4), seed=43)

        # Eager execution
        eager_out = f(x, y)
        
        # Graph execution (tf.function is the closest equivalent to torch.compile)
        compiled_f = tf.function(f)
        compiled_out = compiled_f(x, y)

        # Verify results match
        self.assertTrue(np.allclose(eager_out.numpy(), compiled_out.numpy()))
        
        # Verify numerical correctness
        expected = (x.numpy() + y.numpy()) + 1
        self.assertTrue(np.allclose(eager_out.numpy(), expected))

if __name__ == '__main__':
    unittest.main()