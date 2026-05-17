import torch
import tensorflow as tf
import numpy as np

# Ensure we are in Graph mode as QueueRunner is a v1 API
tf.compat.v1.disable_eager_execution()

# Constants matching the PyTorch example
# 2**20 float32 elements ~ 4 MB
TENSOR_SIZE = 2**20
DTYPE = tf.float32

# Define the graph
with tf.compat.v1.Graph().as_default():
    # Create a FIFO queue to hold the tensors
    # This acts as the buffer for the "custom operation" outputs
    queue = tf.compat.v1.queue.FIFOQueue(capacity=10, dtypes=[DTYPE], shapes=[TENSOR_SIZE])

    # Define a custom operation that generates large tensors
    # This mimics the 'MyOp' in the PyTorch example which creates out_0 and out_1
    def my_enqueue_op():
        # Create large tensors similar to torch.zeros
        out_0 = tf.zeros([TENSOR_SIZE], dtype=DTYPE)
        out_1 = tf.zeros([TENSOR_SIZE], dtype=DTYPE)
        
        # In the PyTorch bug, tensors are saved for backward.
        # Here, we enqueue them into the queue managed by the QueueRunner.
        # We enqueue out_0 to simulate the output being passed along.
        return queue.enqueue(out_0)

    # Create the QueueRunner
    # The QueueRunner manages the threads that run the enqueue operations.
    # This is analogous to the checkpoint manager managing the forward pass.
    qr = tf.compat.v1.train.QueueRunner(queue, [my_enqueue_op()])

    # Dequeue operation to consume the queue and simulate forward pass execution
    dequeue_op = queue.dequeue()

    # Coordinator to manage threads
    coord = tf.compat.v1.train.Coordinator()

    with tf.compat.v1.Session() as sess:
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(queue.initializer)

        # Create threads
        threads = qr.create_threads(sess, coord=coord, start=True)

        print("Starting memory leak test loop...")

        for i in range(100):
            try:
                # Perform a step (dequeue)
                # This ensures the queue runner is active and tensors are being generated
                sess.run(dequeue_op)
            except tf.errors.OutOfRangeError:
                break

            # In the PyTorch bug, the checkpoint mechanism raises an exception to exit early.
            # Here, we simulate the "end of the region" or "early stopping" by requesting 
            # the runner to stop. This tests if the resources (tensors/threads) are properly 
            # cleaned up when the execution flow is interrupted or stopped.
            coord.request_stop()
            coord.join(threads)

            # Check memory usage
            # Note: TF memory allocator behavior differs from PyTorch.
            # We look at 'current' allocation to detect leaks.
            try:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                print(f"Iteration {i}: {mem_info['current'] / 1024**2:.2f} MiB")
            except (RuntimeError, ValueError) as e:
                # Fallback if GPU is not available or memory tracking fails
                print(f"Iteration {i}: Memory check skipped (Error: {e})")

            # Restart the runner for the next iteration
            # This mimics the repeated calls to checkpoint in the loop
            coord = tf.compat.v1.train.Coordinator()
            threads = qr.create_threads(sess, coord=coord, start=True)

        # Final cleanup
        coord.request_stop()
        coord.join(threads)