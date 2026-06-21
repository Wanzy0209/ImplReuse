import tensorflow as tf

# QueueRunners are not compatible with eager execution, so we must disable it.
tf.compat.v1.disable_eager_execution()

def get_default_queue_runner():
    """Creates a default QueueRunner with a FIFOQueue."""
    # Define a queue
    queue = tf.compat.v1.queue.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[()])
    # Define an enqueue operation using random data (mimicking RNG usage)
    enqueue_op = queue.enqueue(tf.random.uniform(shape=()))
    # Create a QueueRunner with multiple threads
    qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op] * 2)
    return queue, qr

def get_sample_inputs():
    """Generates sample inputs (random tensor)."""
    return tf.random.uniform(shape=())

def main():
    # Setup
    queue, qr = get_default_queue_runner()
    dequeue_op = queue.dequeue()

    # Context: Graph definition (analogous to CUDA Graph capture context)
    with tf.Graph().as_default():
        # Re-initialize inside the graph context
        queue, qr = get_default_queue_runner()
        dequeue_op = queue.dequeue()

        # API Call: add_queue_runner
        # This adds the QueueRunner to the graph collection
        tf.compat.v1.train.add_queue_runner(qr)

        # Execution
        with tf.compat.v1.Session() as sess:
            # Initialize variables (if any, though random ops don't strictly need vars here)
            sess.run(tf.compat.v1.global_variables_initializer())

            # Start the queue runners
            coord = tf.compat.v1.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

            # Run the dequeue operation multiple times
            outputs = []
            for _ in range(5):
                output = sess.run(dequeue_op)
                outputs.append(output)
                print('Dequeued output:', output)

            # Stop the threads
            coord.request_stop()
            coord.join(threads)

            # Verification
            # Ensure we got the expected number of outputs
            assert len(outputs) == 5, "Expected 5 outputs"
            # Ensure outputs are valid floats
            for out in outputs:
                assert isinstance(out, float), "Output should be a float"

if __name__ == "__main__":
    main()