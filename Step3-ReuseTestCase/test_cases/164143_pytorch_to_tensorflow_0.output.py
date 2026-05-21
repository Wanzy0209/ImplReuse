import torch
import tensorflow as tf
import unittest

class TestTPURewriteSilentFailure(unittest.TestCase):
    def test_tpu_rewrite_with_cpu_device_context(self):
        """
        Adapted from PyTorch Issue 164143.
        
        Original Bug: torch.compile silently disables compilation when 
        PyTorch's DebugMode (a dispatch mode) is active.
        
        Adapted Test: Verifies that tf.compat.v1.tpu.rewrite raises an error
        when used in a conflicting context (CPU device) instead of 
        silently disabling compilation or falling back to eager execution.
        """
        # Define a simple computation to be rewritten/compiled
        def computation(x):
            return x * 2

        # In PyTorch, the bug was triggered by entering a 'DebugMode' context.
        # In TensorFlow, a conflicting context for TPU rewrite is explicitly 
        # forcing execution on a CPU device, which contradicts the TPU requirement.
        with tf.device('/CPU:0'):
            # We expect an error because we are trying to rewrite for TPU
            # while explicitly on CPU (and without TPU initialization).
            # If the API silently returned a CPU-executed function or 
            # skipped the rewrite without warning, that would be the bug.
            with self.assertRaises(Exception):
                # tf.compat.v1.tpu.rewrite expects inputs as a list of lists of tensors
                inputs = [[tf.constant([1.0])]]
                tf.compat.v1.tpu.rewrite(computation, inputs=inputs)

if __name__ == '__main__':
    unittest.main()