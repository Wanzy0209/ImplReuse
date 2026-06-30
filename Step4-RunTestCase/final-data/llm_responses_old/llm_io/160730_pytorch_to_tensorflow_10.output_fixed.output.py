import sys
import numpy as np

# Attempt to import dependencies and handle environment issues gracefully
try:
    import torch
    import tensorflow as tf
    # Disable eager execution to use tf.compat.v1 components properly
    tf.compat.v1.disable_eager_execution()
except ImportError as e:
    print(f"Skipping test due to missing dependencies or environment issues: {e}")
    sys.exit(0)

def test_range_input_producer():
    """
    Adapted test case for tf.compat.v1.train.range_input_producer.
    Preserves the logic of verifying correctness (Reference vs Actual) 
    to detect potential silent computation errors.
    """
    # Setup parameters
    limit = 10
    num_epochs = 2
    batch_size = 5
    np.random.seed(0)

    # Expected behavior (Reference)
    # The producer should output integers 0 to limit-1, repeated num_epochs times
    expected_output = list(range(limit)) * num_epochs

    with tf.compat.v1.Session() as sess:
        # Original API Under Test: tf.compat.v1.train.range_input_producer
        # This produces a queue of integers
        queue = tf.compat.v1.train.range_input_producer(
            limit=limit, 
            num_epochs=num_epochs, 
            shuffle=False,  # Keep deterministic for assertion
            seed=0,
            capacity=32
        )

        # Dequeue elements to verify the output
        dequeue_op = queue.dequeue_many(batch_size)

        # Initialize local variables (required for num_epochs counter)
        sess.run([tf.compat.v1.local_variables_initializer(), 
                  tf.compat.v1.global_variables_initializer()])

        # Start queue runners
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        actual_output = []

        try:
            # Execution: Run the queue to get all elements
            while True:
                try:
                    batch = sess.run(dequeue_op)
                    actual_output.extend(batch)
                except tf.errors.OutOfRangeError:
                    # Expected when queue is exhausted after num_epochs
                    break
        finally:
            coord.request_stop()
            coord.join(threads)

        # Assertion: Verify the actual output matches the expected reference
        # This mimics torch.testing.assert_close(eager_res, compile_res)
        try:
            np.testing.assert_array_equal(actual_output, expected_output)
            print("Test passed: No silent computation error detected.")
        except AssertionError as e:
            print(f"Test failed: Silent computation error detected.\n{e}")
            raise

if __name__ == "__main__":
    test_range_input_producer()