import tensorflow as tf
import unittest

class TestAddQueueRunner(unittest.TestCase):
    """
    Adapted test case based on PyTorch Issue 162480.
    
    The original PyTorch issue involved a missing check for float types in 
    `rebind_unbacked`, which caused crashes during AOTInductor compilation.
    The fix added a check: `if isinstance(u1, float): continue`.

    This test adapts that logic to `tf.compat.v1.train.add_queue_runner`.
    We verify that the API handles type inputs correctly, specifically testing
    the boundary condition of passing a float (the type involved in the original bug).
    """

    def setUp(self):
        # QueueRunners require graph mode (TF 1.x behavior)
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

    def test_add_queue_runner_valid_input(self):
        """Test that a valid QueueRunner is added to the collection."""
        with tf.compat.v1.Graph().as_default():
            queue = tf.compat.v1.queue.FIFOQueue(capacity=10, dtypes=[tf.float32])
            enqueue_op = queue.enqueue([1.0])
            qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op])

            # Add the valid runner
            tf.compat.v1.train.add_queue_runner(qr)

            # Verify it is in the collection
            runners = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)
            self.assertIn(qr, runners)
            self.assertEqual(len(runners), 1)

    def test_add_queue_runner_float_handling(self):
        """
        Test handling of float inputs.
        
        In the PyTorch bug, a float value appearing in a symbolic context 
        caused a crash because it wasn't handled. 
        Here, we test if `add_queue_runner` validates input types strictly 
        or handles a float argument gracefully/robustly.
        """
        with tf.compat.v1.Graph().as_default():
            # The PyTorch bug was triggered by a float being processed.
            # We attempt to pass a float to the TensorFlow API.
            invalid_input = 3.14159
            
            # TensorFlow APIs are generally type-strict. We expect a TypeError or ValueError.
            # This verifies that the API does not silently fail or crash ambiguously
            # when encountering the type that caused issues in PyTorch.
            with self.assertRaises((TypeError, ValueError)):
                tf.compat.v1.train.add_queue_runner(invalid_input)

if __name__ == "__main__":
    unittest.main()