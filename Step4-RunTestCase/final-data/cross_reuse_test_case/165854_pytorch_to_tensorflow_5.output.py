import sys

try:
    import torch
    import tensorflow as tf
except ImportError as e:
    # Handle environment dependency issues (e.g., GLIBCXX version mismatch)
    if "GLIBCXX" in str(e) or "libstdc++" in str(e):
        print("Skipping test: Environment dependency error detected.")
        print(f"Error details: {e}")
        print("This usually indicates a system library version mismatch (e.g., libstdc++.so.6).")
        sys.exit(0)
    else:
        # If it's a different import error, raise it normally
        raise

def run_with_capacity(capacity, collection):
    """Run queue runner setup with a specific capacity, creating a queue sized by capacity."""
    # Create a queue that depends on dynamic capacity
    # This mimics the 'head_scale' buffer creation in the PyTorch example
    queue = tf.compat.v1.FIFOQueue(capacity=capacity, dtypes=[tf.float32], shapes=[])

    # Define an enqueue operation
    enqueue_op = queue.enqueue([tf.random.uniform([])])

    # Create a QueueRunner
    qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op])

    print(f"  Running with capacity={capacity}, queue={queue.name}")

    # The API under test: add_queue_runner
    # This adds the runner to the graph collection, similar to how torch.compile captures state
    tf.compat.v1.train.add_queue_runner(qr, collection=collection)

    return queue

def main():
    # Disable eager execution to use TF 1.x graph mode required for this API
    tf.compat.v1.disable_eager_execution()
    
    # Test with different capacities - this makes capacity a dynamic dimension
    # and the captured resource (queue) changes size with capacity
    capacities = [4, 8, 4, 16, 4]

    with tf.compat.v1.Graph().as_default():
        queues = []
        collection = tf.compat.v1.GraphKeys.QUEUE_RUNNERS

        print(f"Running queue runner test with dynamic capacities")
        print(f"Testing capacities: {capacities}\n")

        for iteration, cap in enumerate(capacities, start=1):
            print(f"Iteration {iteration}:")
            q = run_with_capacity(cap, collection)
            queues.append(q)

        # Verify execution by running the session
        with tf.compat.v1.Session() as sess:
            # Initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())

            # Start all queue runners added to the collection
            coord = tf.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

            # Attempt to dequeue from the queues to verify they are active
            # This mimics the forward/backward pass execution in the PyTorch example
            for i, q in enumerate(queues):
                print(f"  Verifying queue {i} (capacity {capacities[i]}):")
                try:
                    for _ in range(2):
                        val = sess.run(q.dequeue())
                        print(f"    Dequeued value: {val}")
                except tf.errors.OutOfRangeError:
                    print("    Queue empty/finished")

            coord.request_stop()
            coord.join(threads)

        print(f"\n Completed all iterations")

if __name__ == "__main__":
    main()