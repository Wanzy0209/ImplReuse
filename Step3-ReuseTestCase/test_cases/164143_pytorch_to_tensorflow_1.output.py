import torch
import tensorflow as tf
import unittest

class TestBatchParallelDebugMode(unittest.TestCase):
    def test_batch_parallel_with_unsupported_context(self):
        """
        Adapted from PyTorch Issue 164143.
        
        Original Bug: DebugMode (a dispatch mode) causes torch.compile to silently 
        skip compilation instead of raising an error.
        
        TensorFlow Adaptation: Verify that tf.compat.v1.tpu.batch_parallel raises 
        an error when executed in an incompatible context (e.g., CPU), rather than 
        silently falling back to CPU execution (which would be the equivalent of 
        "disabling" the TPU parallelization).
        """
        
        # Define a simple computation to be parallelized
        def computation(x):
            return x + 1

        # Wrap the call in a function to trigger the compilation/execution graph
        @tf.function
        def run_on_cpu():
            # Force execution on CPU, which is an unsupported mode for batch_parallel
            with tf.device("/CPU:0"):
                return tf.compat.v1.tpu.batch_parallel(
                    computation,
                    inputs=[[1.0, 2.0]],
                    num_shards=1
                )

        # Expected behavior: Raise an error because TPU is required but not available/context is wrong.
        # If it runs successfully on CPU, it implies the API silently disabled the TPU optimization (the bug).
        with self.assertRaises(Exception) as context:
            run_on_cpu()

        # Verify the error is related to the device/TPU context
        error_message = str(context.exception).lower()
        self.assertTrue(
            "tpu" in error_message or "device" in error_message, 
            f"Expected TPU/Device error, but got: {context.exception}"
        )

if __name__ == "__main__":
    unittest.main()