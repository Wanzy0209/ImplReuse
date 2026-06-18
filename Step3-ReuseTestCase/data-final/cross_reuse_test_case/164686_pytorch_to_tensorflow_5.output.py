import torch
import tensorflow as tf
import numpy as np

# Disable eager execution to use tf.compat.v1 graph mode (required for the API under test)
tf.compat.v1.disable_eager_execution()

# Set seed for reproducibility matching the original bug report
np.random.seed(13653)

# Generate inputs similar to the PyTorch fuzzer
# PyTorch: torch.tensor(torch.randn(()), dtype=torch.int64).item()
arg_0 = int(np.random.randn())
arg_1 = int(np.random.randn())

# Define the graph
with tf.compat.v1.Graph().as_default():
    # Placeholders for inputs
    p_arg_0 = tf.compat.v1.placeholder(dtype=tf.int64, name="arg_0")
    p_arg_1 = tf.compat.v1.placeholder(dtype=tf.int64, name="arg_1")

    # Replicate the arithmetic logic from the bug report to stress type handling
    # var_node_3 = torch.full((), 1.0, dtype=torch.float32)
    var_node_3 = tf.constant(1.0, dtype=tf.float32)

    # var_node_5 = -3 (int32)
    var_node_5 = tf.constant(-3, dtype=tf.int32)

    # var_node_4 = var_node_5 + var_node_6 (int32)
    # Note: var_node_6 is arg_0 (int64). Cast to int32 for addition.
    var_node_4 = tf.add(var_node_5, tf.cast(p_arg_0, tf.int32))

    # var_node_1 = var_node_2 + var_node_4 (float32)
    # var_node_2 is var_node_3.item() (float32).
    var_node_1 = tf.add(var_node_3, tf.cast(var_node_4, tf.float32))

    # var_node_9 = 1 (int64)
    var_node_9 = tf.constant(1, dtype=tf.int64)

    # var_node_10 = -10 (int32)
    var_node_10 = tf.constant(-10, dtype=tf.int32)

    # var_node_8 = var_node_9 / var_node_10 (int64)
    # Using floor division and casting to match the specific dtype comment in the bug
    var_node_8 = tf.cast(tf.floordiv(var_node_9, tf.cast(var_node_10, tf.int64)), tf.int64)

    # var_node_13 = -5 (int32)
    var_node_13 = tf.constant(-5, dtype=tf.int32)

    # var_node_11 = var_node_12 / var_node_13 (int32)
    # var_node_12 is arg_1 (int64).
    var_node_11 = tf.cast(tf.floordiv(p_arg_1, tf.cast(var_node_13, tf.int64)), tf.int32)

    # var_node_7 = var_node_8 + var_node_11 (int32)
    var_node_7 = tf.add(tf.cast(var_node_8, tf.int32), var_node_11)

    # var_node_0 = var_node_1 * var_node_7 (float32)
    var_node_0 = tf.multiply(var_node_1, tf.cast(var_node_7, tf.float32))

    # Sentinel logic (simplified for TF graph)
    sentinel = tf.constant(1.0, dtype=tf.float32)
    result = tf.multiply(var_node_0, sentinel)

    # Setup Queue and QueueRunner to test the target API
    queue = tf.compat.v1.queue.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[])
    enqueue_op = queue.enqueue(result)

    # Create the QueueRunner
    qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op])

    # --- API Under Test ---
    tf.compat.v1.train.add_queue_runner(qr)
    # ---------------------

    # Verify the runner was added to the collection
    runners = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)
    assert qr in runners, "QueueRunner was not added to the collection."

    # Calculate expected value using numpy (Eager mode equivalent)
    def eager_logic(a0, a1):
        v3 = 1.0
        v5 = -3
        v4 = v5 + int(a0) # int32
        v1 = v3 + float(v4) # float32
        v9 = 1
        v10 = -10
        v8 = v9 // v10 # int64
        v13 = -5
        v11 = int(a1) // v13 # int32
        v7 = v8 + v11 # int32
        v0 = v1 * v7 # float32
        return v0 * 1.0

    expected_val = eager_logic(arg_0, arg_1)

    # Run the session
    with tf.compat.v1.Session() as sess:
        # Initialize variables (if any) and local variables (for queues)
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        # Start the queue runner threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # Dequeue the result
        dequeued_val = sess.run(queue.dequeue(), feed_dict={p_arg_0: arg_0, p_arg_1: arg_1})

        # Check for divergence (Graph vs Eager)
        # Using np.isclose because of float arithmetic
        if np.isclose(dequeued_val, expected_val):
            print(" Test passed: Graph and Eager results match.")
        else:
            print(f" Test failed: Divergence detected. Graph={dequeued_val}, Eager={expected_val}")

        # Stop threads
        coord.request_stop()
        coord.join(threads)