import tensorflow as tf
import unittest

class TestOutputAllIntermediates(unittest.TestCase):
    """
    Test case for tf.compat.v1.experimental.output_all_intermediates.
    
    This test is derived from the logic of Issue #164491, which highlights 
    unexpected behavior (slowness/errors) in experimental APIs under specific 
    input configurations (row-major matrices). 
    
    Here, we verify that the experimental API `output_all_intermediates` 
    behaves correctly according to its documented conditions (e.g., eager mode 
    vs graph mode), ensuring it respects the "guards" mentioned in its 
    docstring.
    """

    def test_return_type(self):
        """Verify that the API returns a boolean value."""
        result = tf.compat.v1.experimental.output_all_intermediates()
        self.assertIsInstance(result, bool)

    def test_eager_mode_behavior(self):
        """
        Test behavior in eager mode.
        
        The docstring states: "guards against outputting intermediates in eager mode".
        We expect the return value to be False in this context.
        """
        # Ensure we are in eager mode
        self.assertTrue(tf.executing_eagerly())
        
        result = tf.compat.v1.experimental.output_all_intermediates()
        self.assertFalse(result, 
            "output_all_intermediates should return False in eager mode "
            "to guard against unnecessary output.")

    def test_tf_function_behavior(self):
        """
        Test behavior inside a tf.function.
        
        The docstring states: "do not output intermediates of tf.function".
        We expect the return value to be False when called within a tf.function context.
        """
        @tf.function
        def check_in_function():
            return tf.compat.v1.experimental.output_all_intermediates()

        result = check_in_function()
        self.assertFalse(result,
            "output_all_intermediates should return False inside a tf.function.")

if __name__ == '__main__':
    unittest.main()