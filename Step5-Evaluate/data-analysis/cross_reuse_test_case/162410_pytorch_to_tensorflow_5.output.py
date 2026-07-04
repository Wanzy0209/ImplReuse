import tensorflow as tf
import numpy as np

# Disable eager execution as QueueRunners are designed for graph mode
tf.compat.v1.disable_eager_execution()

def test_queue_runner_with_tensor_ops():
    """
    Adapts the PyTorch test case logic to TensorFlow using tf.compat.v1.train.add_queue_runner.
    
    Original Logic:
    1. x.copy_(x.flip(1))  -> In-place modification
    2. y = y.sum(dim=1, keepdim=True) + y
    3. return x + y
    
    TensorFlow Adaptation:
    - Uses a QueueRunner to manage the input data flow (mimicking the setup/execution phase).
    - Uses tf.Variable to simulate the in-place modification of 'x'.
    - Verifies the graph execution result against a NumPy reference.
    """
    
    # Setup data (matching original dimensions)
    # Using smaller size for quick testing, but logic holds for large tensors
    batch_size = 20
    feature_size = 1024 * 10 # Reduced slightly for faster execution in this example
    
    np_x = np.random.randn(batch_size, feature_size).astype(np.float32)
    np_y = np.random.randn(batch_size, feature_size).astype(np.float32)
    
    # Reference calculation (Eager/NumPy equivalent of the PyTorch function)
    # x.copy_(x.flip(1))
    ref_x = np_x.copy()
    ref_x = np.flip(ref_x, axis=1)
    # y = y.sum(dim=1, keepdim=True) + y
    ref_y_sum = np.sum(np_y, axis=1, keepdims=True)
    ref_y = ref_y_sum + np_y
    # return x + y
    ref_result = ref_x + ref_y

    # Graph construction
    with tf.compat.v1.Session() as sess:
        # FIX: Use tf.constant instead of placeholders.
        # Placeholders require a feed_dict when run, but QueueRunner threads
        # execute operations independently in the background and cannot receive
        # the feed_dict passed in the main thread. Using constants embeds the data
        # directly in the graph, allowing the QueueRunner to enqueue it successfully.
        x_data = tf.constant(np_x)
        y_data = tf.constant(np_y)

        # FIFO Queue to manage inputs
        q = tf.queue.FIFOQueue(capacity=1, dtypes=[tf.float32, tf.float32], 
                               shapes=[[batch_size, feature_size], [batch_size, feature_size]])

        # Enqueue operation
        enqueue_op = q.enqueue([x_data, y_data])

        # Create QueueRunner
        # This is the API under test: tf.compat.v1.train.add_queue_runner
        qr = tf.compat.v1.train.QueueRunner(q, [enqueue_op])
        tf.compat.v1.train.add_queue_runner(qr)

        # Dequeue data for processing
        x_dequeued, y_dequeued = q.dequeue()

        # Variable to simulate the mutable tensor 'x' in PyTorch
        x_var = tf.Variable(tf.zeros_like(x_dequeued), dtype=tf.float32)

        # Initialize x_var with the dequeued data
        init_x = x_var.assign(x_dequeued)

        # Operation 1: x.copy_(x.flip(1))
        # In TF, we assign the reversed value back to the variable
        with tf.control_dependencies([init_x]):
            flipped_x = tf.reverse(x_var, axis=[1])
            update_x = x_var.assign(flipped_x)

        # Operation 2: y = y.sum(dim=1, keepdim=True) + y
        y_sum = tf.reduce_sum(y_dequeued, axis=1, keepdims=True)
        y_new = y_sum + y_dequeued

        # Operation 3: return x + y
        # Ensure update_x happens before the addition
        with tf.control_dependencies([update_x]):
            result = x_var + y_new

        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())

        # Start QueueRunners
        coord = tf.train.Coordinator()
        threads = qr.create_threads(sess, coord=coord, start=True)

        # The QueueRunner is now responsible for enqueuing the data.
        # We no longer need to manually run sess.run(enqueue_op, feed_dict=...).
        # The runner will enqueue the constants, blocking when the queue is full.
        # The sess.run(result) below will dequeue, unblocking the runner.

        # Execute the graph and fetch result
        tf_result = sess.run(result)

        # Verification
        try:
            np.testing.assert_allclose(tf_result, ref_result, rtol=1e-5, atol=1e-5)
            print("Test passed: TensorFlow QueueRunner execution matches reference logic.")
        except AssertionError as e:
            print("Test failed: Numerical mismatch detected.")
            raise e
        finally:
            # Cleanup
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_queue_runner_with_tensor_ops()