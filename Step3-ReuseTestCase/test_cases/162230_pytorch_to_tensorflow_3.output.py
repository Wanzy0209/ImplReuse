import unittest
import tensorflow as tf
import time

class TestHeNormalSimilar(unittest.TestCase):
    def test_he_normal_initialization_timeout(self):
        """
        Adapted from the PyTorch test_scalar_multiply which failed due to a 
        subprocess.TimeoutExpired (30s). This test verifies that 
        tf.keras.initializers.HeNormal executes within a reasonable time limit 
        when used in a graph context (tf.function), mimicking the graph 
        execution scenario of the original bug.
        """
        
        # The original test timed out after 30 seconds.
        TIMEOUT_SECONDS = 30
        
        @tf.function
        def run_initializer(shape):
            # Target API: tf.keras.initializers.HeNormal
            initializer = tf.keras.initializers.HeNormal()
            return initializer(shape)

        start_time = time.time()
        
        # Perform the operation
        # Using a shape that might trigger graph compilation complexity
        tensor = run_initializer((1000, 1000))
        
        # Ensure the tensor is materialized
        _ = tensor.numpy()
        
        elapsed = time.time() - start_time
        
        # Assert completion within timeout
        self.assertLess(elapsed, TIMEOUT_SECONDS, 
                         f"Test timed out after {elapsed:.2f} seconds")
        
        # Basic sanity check on the output
        self.assertEqual(tensor.shape, (1000, 1000))

if __name__ == '__main__':
    unittest.main()