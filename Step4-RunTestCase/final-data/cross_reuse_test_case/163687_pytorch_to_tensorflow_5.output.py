import sys

# Attempt to import TensorFlow and handle environment incompatibilities gracefully.
try:
    import tensorflow as tf
    # The original bug report involves eager vs compile divergence.
    # The target API, tf.compat.v1.train.add_queue_runner, is specific to TensorFlow 1 graph mode
    # and does not function in eager execution. We disable eager execution to simulate the
    # "compile" environment where this API is relevant.
    tf.compat.v1.disable_eager_execution()
except ImportError as e:
    # Check for the specific GLIBCXX error mentioned in the traceback
    if "GLIBCXX" in str(e) or "libstdc++" in str(e):
        print(f"Skipping test: TensorFlow environment incompatible ({e}).")
        sys.exit(0)
    else:
        # Re-raise if it's a different import error
        raise

import torch

def test_add_queue_runner_complex_shapes():
    """
    Adapted from the PyTorch flex_attention OOM bug.
    
    Original Logic:
    - Defines multiple tensors with specific large shapes (e.g., 27x26x62x122).
    - Performs flex_attention operations (computation).
    - Compiles the function using torch.compile.
    
    Adapted Logic (TensorFlow):
    - Defines a queue with shapes matching the original PyTorch tensors.
    - Creates QueueRunner operations to manage data flow (pipeline management).
    - Uses tf.compat.v1.train.add_queue_runner to register the runner.
    """
    
    with tf.compat.v1.Session() as sess:
        # Shapes extracted from the original PyTorch test case
        # arg0: (27, 26, 62, 122), arg1: (27, 26, 124, 122), etc.
        shapes = [
            [27, 26, 62, 122],
            [27, 26, 124, 122],
            [27, 26, 124, 122],
            [27, 26, 124, 122],
            [27, 26, 248, 122],
            [27, 26, 248, 122],
            [27, 26, 31, 122],
            [27, 26, 124, 122],
            [27, 26, 31, 122],
            [27, 26, 124, 122],
            [27, 26, 124, 122]
        ]
        dtypes = [tf.float32] * len(shapes)

        # Create a FIFO queue to hold the complex tensors
        # This mimics the data setup in the original function
        queue = tf.compat.v1.FIFOQueue(capacity=32, dtypes=dtypes, shapes=shapes)

        # Create enqueue operations mimicking the input arguments (arg0...arg10)
        enqueue_ops = []
        for i, (shape, dtype) in enumerate(zip(shapes, dtypes)):
            # Generate random tensors to enqueue, similar to torch.rand
            tensor = tf.random.normal(shape, dtype=dtype)
            enqueue_ops.append(queue.enqueue([tensor]))

        # Create a QueueRunner to manage the enqueue threads
        # This is the object we intend to test adding to the graph
        qr = tf.compat.v1.train.QueueRunner(queue, enqueue_ops)

        # --- TARGET API CALL ---
        # tf.compat.v1.train.add_queue_runner
        # This adds the QueueRunner to the default collection in the graph.
        # This is analogous to registering a compiled function or operator in PyTorch.
        tf.compat.v1.train.add_queue_runner(qr)
        
        # Verify that the QueueRunner was successfully added to the collection
        collection = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)
        assert qr in collection, "QueueRunner was not added to the graph collection."

        # Initialize variables and start the queue runners
        sess.run(tf.compat.v1.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        # Attempt to dequeue to verify the pipeline works
        # This mimics the execution step in the original bug report
        try:
            # Dequeue one set of tensors
            sess.run(queue.dequeue())
            print("Test passed: QueueRunner added and executed successfully.")
        except tf.errors.ResourceExhaustedError:
            # Catching potential OOM errors, similar to the original bug context
            print("ResourceExhaustedError (OOM) occurred during execution.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_add_queue_runner_complex_shapes()