import torch
import tensorflow as tf
import unittest

class TestStringInputProducer(unittest.TestCase):
    """
    Adapted from PyTorch Issue 164143: DebugMode silently disables torch.compile.
    
    The original bug describes a scenario where a specific execution mode (DebugMode)
    causes an API (torch.compile) to silently skip its work instead of raising an error.
    
    This test verifies the behavior of the similar TensorFlow API 
    (tf.compat.v1.train.string_input_producer) when used in an incompatible 
    execution mode (Eager Execution vs Graph Execution). We assert that it 
    raises an explicit error rather than failing silently or returning a 
    non-functional object.
    """

    def test_string_input_producer_in_eager_mode(self):
        """
        Test that string_input_producer raises an error in Eager mode.
        
        In PyTorch, the bug was a silent skip. Here, we verify that TF
        explicitly fails when the API is used outside its intended 
        Graph/Session context.
        """
        # Ensure we are in eager execution mode (TF2 default)
        # This acts as the "incompatible mode" analogous to PyTorch's DebugMode
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()

        filenames = ["file1.txt", "file2.txt"]

        # The API expects to be run in a graph context with a Session.
        # In eager mode, it should raise an error (e.g., regarding graph/queue mechanics).
        # We assert that an exception is raised to prevent "silent" failures.
        with self.assertRaises(Exception) as context:
            tf.compat.v1.train.string_input_producer(filenames, num_epochs=1)

        # Optional: Check that the error message is relevant to the context mismatch
        self.assertTrue("graph" in str(context.exception).lower() or 
                        "session" in str(context.exception).lower() or
                        "not supported" in str(context.exception).lower())

if __name__ == '__main__':
    unittest.main()