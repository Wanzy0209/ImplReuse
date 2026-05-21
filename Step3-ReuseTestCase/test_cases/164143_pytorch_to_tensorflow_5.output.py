import torch
import tensorflow as tf
import unittest

class TestAddQueueRunnerCompatibility(unittest.TestCase):
    """
    Adapted from PyTorch Issue 164143 (DebugMode silently disables torch.compile).
    
    This test verifies the behavior of tf.compat.v1.train.add_queue_runner
    when used in an incompatible execution mode (Eager Execution).
    
    Original Bug Logic:
    - API: torch.compile
    - Incompatible Mode: DebugMode
    - Behavior: Silently skips compilation (logs but doesn't error).
    
    Adapted Test Logic:
    - API: tf.compat.v1.train.add_queue_runner
    - Incompatible Mode: Eager Execution (TF2 default)
    - Expected Behavior: Should ideally error or warn, but currently may 
      silently accept the registration without functionality.
    """

    def test_add_queue_runner_in_eager_mode(self):
        # Ensure we are in eager execution (TF2 default)
        self.assertTrue(tf.executing_eagerly(), "Test requires eager execution enabled.")

        # Create a simple FIFOQueue and a QueueRunner
        # This mimics the setup required for the API under test
        queue = tf.queue.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[])
        enqueue_op = queue.enqueue([1.0])
        qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op])

        # Attempt to add the queue runner
        # In the PyTorch bug, torch.compile would log "skipping" but return a callable.
        # Here, we check if add_queue_runner raises an error or silently accepts the input.
        
        # Note: tf.compat.v1.train.add_queue_runner calls ops.add_to_collection.
        # In eager mode, this might succeed but the runner won't actually start
        # without a Session.
        try:
            tf.compat.v1.train.add_queue_runner(qr)
            
            # Verify if it was added to the collection
            runners = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)
            
            # If it was added without error in eager mode, this mirrors the "silent" behavior
            # where the system accepts an invalid state.
            self.assertIn(qr, runners, "QueueRunner was not added to collection.")
            
            # This assertion passes if the API silently accepts the incompatible mode,
            # reproducing the core logic of the original bug report (silent failure/acceptance).
            print("API call succeeded in eager mode (Silent behavior detected).")

        except Exception as e:
            # If it raises an error, that is the "fixed" behavior (explicit failure).
            self.fail(f"add_queue_runner raised an explicit error in eager mode: {e}")

if __name__ == "__main__":
    unittest.main()