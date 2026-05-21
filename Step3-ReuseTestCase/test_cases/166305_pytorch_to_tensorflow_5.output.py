import tensorflow as tf

# Setup: TF 1.x graph mode is required for compat.v1.train.add_queue_runner
# Analogous to dynamo.config.optimize_ddp = True in the PyTorch repro
tf.compat.v1.disable_eager_execution()

# Define a simple queue and enqueue operation
# Analogous to defining the custom autograd function logic
queue = tf.compat.v1.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[()])
enqueue_op = queue.enqueue([1.0])

# Create a QueueRunner
# Analogous to the DoubleLayer or Model definition
qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op] * 2)

# Add the QueueRunner to the graph collection
# This is the API under test: tf.compat.v1.train.add_queue_runner
# Analogous to wrapping the model in DDP
tf.compat.v1.train.add_queue_runner(qr)

# Define the dequeue operation to consume data
dequeue_op = queue.dequeue()

def main():
    with tf.compat.v1.Session() as sess:
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        # Coordinator for managing threads
        coord = tf.compat.v1.train.Coordinator()

        # Start the queue runners (which includes the one we added)
        # Analogous to starting the training loop
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        try:
            # Run a few iterations
            # Analogous to "for it in range(3)"
            for i in range(3):
                val = sess.run(dequeue_op)
                print(f"Iteration {i+1}: value={val:.1f}")
        finally:
            # Stop the threads
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    main()