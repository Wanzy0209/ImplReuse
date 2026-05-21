import torch
import tensorflow as tf
import threading

# Disable eager execution to use TF1-style graph mode required for QueueRunners
tf.compat.v1.disable_eager_execution()

def tf_standard_add(x: tf.Tensor, y: tf.Tensor):
    """Standard TensorFlow operation, analogous to torch_add."""
    return x + y

def tf_queue_runner_add(q: tf.Queue, qr: tf.compat.v1.train.QueueRunner):
    """
    Calls the API under test: tf.compat.v1.train.add_queue_runner.
    This adds the QueueRunner to the graph collection.
    """
    return tf.compat.v1.train.add_queue_runner(qr, collection=tf.compat.v1.GraphKeys.QUEUE_RUNNERS)

def main():
    with tf.compat.v1.Session() as sess:
        # Setup standard operation
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.constant([4.0, 5.0, 6.0])
        z = tf_standard_add(x, y)

        # Setup Queue and QueueRunner
        # We use a FIFOQueue to mimic the data processing aspect
        q = tf.compat.v1.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[()])
        enqueue_op = q.enqueue([1.0])

        # Reproduce the loop structure from the original PyTorch script
        # The original script loops to check GIL behavior over multiple calls.
        for i in range(10):
            # Create a new QueueRunner for each iteration to test the API call repeatedly
            # In a real scenario, one might reuse runners, but this stresses the API registration.
            qr = tf.compat.v1.train.QueueRunner(q, [enqueue_op])
            
            # Call the API under test
            tf_queue_runner_add(q, qr)
            
            # Execute standard op to mix graph execution with API calls
            sess.run(z)

        # Verification: Check that the runners were successfully added to the collection
        runners = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)
        print(f"Total runners added to collection: {len(runners)}")
        assert len(runners) == 10, f"Expected 10 runners, but found {len(runners)}"

        # Initialize queue and variables
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(q.initializer)

        # Start the queue runners to ensure they are functional
        # This step verifies that the runners added by the API work correctly.
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        # Perform a dequeue to verify the pipeline is active
        # This mimics the data consumption aspect
        for _ in range(5):
            result = sess.run(q.dequeue())
            print(f"Dequeued value: {result}")

        # Cleanup
        coord.request_stop()
        coord.join(threads)
        print("Test completed successfully.")

if __name__ == "__main__":
    main()