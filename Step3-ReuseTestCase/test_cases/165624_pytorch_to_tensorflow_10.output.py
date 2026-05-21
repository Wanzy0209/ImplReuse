import tensorflow as tf
import sys

# The original bug involves a conditional block being executed twice due to a merge mistake.
# This test verifies that tf.compat.v1.train.range_input_producer does not suffer from a similar
# logic error where the production loop runs twice when configured to run once.

# Disable eager execution to use TF1 queue runners
tf.compat.v1.disable_eager_execution()

def test_range_input_producer_no_double_execution():
    """
    Verifies that range_input_producer produces the correct number of items
    and does not duplicate the production logic (similar to the double execution bug).
    """
    limit = 10
    num_epochs = 1
    
    with tf.compat.v1.Session() as sess:
        # Configure the producer to run exactly once
        range_producer = tf.compat.v1.train.range_input_producer(
            limit=limit, 
            num_epochs=num_epochs, 
            shuffle=False,
            capacity=32
        )
        
        dequeue_op = range_producer.dequeue()
        
        # Initialize local variables (required for num_epochs)
        sess.run([tf.compat.v1.local_variables_initializer()])
        
        # Start queue runners
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        
        results = []
        try:
            # Attempt to dequeue more items than should exist
            # If there is a "double execution" bug, we might get 2*limit items
            for _ in range(limit * 2):
                val = sess.run(dequeue_op)
                results.append(val)
        except tf.errors.OutOfRangeError:
            # Expected when the queue is empty after num_epochs
            pass
        finally:
            coord.request_stop()
            coord.join(threads)
    
    # Assertions to verify correct behavior (no double execution)
    assert len(results) == limit, \
        f"Bug detected: Expected {limit} items (single execution), but got {len(results)}. " \
        "This suggests a potential double execution logic similar to the PyTorch merge mistake."
    
    assert sorted(results) == list(range(limit)), \
        f"Bug detected: Items do not match expected range. Got {results}"
    
    print("Test passed: No double execution detected.")

if __name__ == "__main__":
    test_range_input_producer_no_double_execution()