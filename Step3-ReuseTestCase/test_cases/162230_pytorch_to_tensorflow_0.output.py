import tensorflow as tf
import time
import unittest

class TestTfStringsAsString(unittest.TestCase):
    """
    Adapted from the PyTorch test_scalar_multiply case.
    Original Bug: The test timed out after 30 seconds during execution.
    This test verifies that tf.strings.as_string executes correctly without hanging.
    """

    def test_as_string_conversion(self):
        # Test 1: Basic scalar conversion (mimicking scalar input from original test)
        scalar_input = tf.constant(42.0)
        result = tf.strings.as_string(scalar_input)
        self.assertEqual(result, b'42.0')

        # Test 2: Basic tensor conversion
        tensor_input = tf.constant([1.5, 2.5, 3.5])
        result = tf.strings.as_string(tensor_input)
        self.assertAllEqual(result, [b'1.5', b'2.5', b'3.5'])

        # Test 3: Precision handling
        tensor_input = tf.constant([3.14159])
        result = tf.strings.as_string(tensor_input, precision=2)
        self.assertEqual(result, b'3.14')

        # Test 4: Scientific notation
        tensor_input = tf.constant([1000.0])
        result = tf.strings.as_string(tensor_input, scientific=True, precision=2)
        self.assertEqual(result, b'1.00e+03')

        # Test 5: Stress test to prevent timeout/hang (Core bug reproduction logic)
        # The original test failed due to a timeout. We verify this operation
        # completes efficiently on a larger dataset.
        large_tensor = tf.random.uniform((5000, 100), minval=0, maxval=1000)
        
        start_time = time.time()
        # Execute the operation
        _ = tf.strings.as_string(large_tensor)
        end_time = time.time()
        
        # Assert completion within a reasonable time (e.g., 5 seconds)
        # to ensure we don't reproduce the timeout issue.
        self.assertLess(end_time - start_time, 5.0, 
                        "tf.strings.as_string operation timed out or took too long")

if __name__ == '__main__':
    unittest.main()