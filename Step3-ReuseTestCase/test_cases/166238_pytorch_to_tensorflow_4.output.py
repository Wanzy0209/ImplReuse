import tensorflow as tf
import unittest

# Disable eager execution to use TF v1 queue-based APIs
tf.compat.v1.disable_eager_execution()

class TestStringInputProducer(unittest.TestCase):
    """
    Adapted test case for tf.compat.v1.train.string_input_producer.
    The original PyTorch test (test_pytree_tree_map_dict_order_cxx) focused on 
    the handling of data structures and order preservation. 
    This test verifies the similar behavior in the TensorFlow API, specifically
    ensuring that the string_input_producer correctly handles the input tensor
    and preserves order when shuffle is disabled.
    """

    def test_string_input_producer_order_preservation(self):
        """
        Verifies that string_input_producer preserves the order of the input tensor
        when shuffle=False. This mirrors the 'dict_order' aspect of the original test.
        """
        with tf.compat.v1.Session() as sess:
            # Define input data
            input_strings = [b"file1.txt", b"file2.txt", b"file3.txt"]
            string_tensor = tf.constant(input_strings)

            # Create the string input producer
            # shuffle=False is critical for testing order preservation
            queue = tf.compat.v1.train.string_input_producer(
                string_tensor, 
                shuffle=False, 
                capacity=10,
                name="test_queue"
            )
            
            # Dequeue elements to verify output
            dequeue_op = queue.dequeue()

            # Initialize local variables (required for num_epochs if used, good practice)
            sess.run(tf.compat.v1.local_variables_initializer())
            
            # Start queue runners
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                results = []
                # Dequeue all items
                for _ in range(len(input_strings)):
                    results.append(sess.run(dequeue_op))
                
                # Assert that the order matches the input
                self.assertEqual(results, input_strings, 
                                 "Order mismatch: string_input_producer did not preserve order with shuffle=False")
            finally:
                coord.request_stop()
                coord.join(threads)

    def test_string_input_producer_epochs(self):
        """
        Verifies the num_epochs functionality to ensure correct cycling behavior,
        analogous to checking the robustness of data structure handling.
        """
        with tf.compat.v1.Session() as sess:
            input_strings = [b"a", b"b"]
            string_tensor = tf.constant(input_strings)

            # Set num_epochs to 2
            queue = tf.compat.v1.train.string_input_producer(
                string_tensor, 
                num_epochs=2, 
                shuffle=False, 
                capacity=10
            )
            dequeue_op = queue.dequeue()

            sess.run(tf.compat.v1.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                results = []
                # Expect 2 epochs * 2 items = 4 items total
                for _ in range(4):
                    results.append(sess.run(dequeue_op))
                
                expected = input_strings * 2
                self.assertEqual(results, expected, 
                                 "Mismatch in num_epochs cycling behavior")
            finally:
                coord.request_stop()
                coord.join(threads)

if __name__ == "__main__":
    unittest.main()