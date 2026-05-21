import sys
import tensorflow as tf

# Preserve the setup from the original test case
sys.setrecursionlimit(10000000)

# The target API (tf.compat.v1.train.add_queue_runner) requires graph mode
tf.compat.v1.disable_eager_execution()

def test_add_queue_runner():
    """
    Adapted test case for tf.compat.v1.train.add_queue_runner.
    Preserves the structure of setting a system limit before execution,
    though the specific recursion limit behavior is specific to the PyTorch bug.
    """
    with tf.compat.v1.Session() as sess:
        # Create a simple FIFO queue
        queue = tf.compat.v1.FIFOQueue(capacity=10, dtypes=[tf.int32])
        
        # Define an enqueue operation
        enqueue_op = queue.enqueue([1])
        
        # Create a QueueRunner
        qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op])
        
        # Call the API under test
        tf.compat.v1.train.add_queue_runner(qr)
        
        # Verify the runner was added and works by starting the queue runners
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        
        # Dequeue an item to verify the pipeline is active
        result = sess.run(queue.dequeue())
        assert result == 1, "Queue did not return the expected value"
        
        # Clean up
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    test_add_queue_runner()
    print("Test passed.")