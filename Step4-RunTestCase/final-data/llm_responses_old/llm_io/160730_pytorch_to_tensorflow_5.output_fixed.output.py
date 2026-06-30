import sys
import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: TensorFlow import failed due to environment issues (e.g., GLIBC version). Error: {e}")
    sys.exit(0)

def test_tf_queue_runner_math_ops():
    """
    Adapts the PyTorch torch.compile test case to TensorFlow using 
    tf.compat.v1.train.add_queue_runner to verify graph execution behavior
    with trigonometric operations and control flow.
    """
    
    # 1. Eager Execution (Baseline)
    # This corresponds to the eager_res = foo(torch.from_numpy(x)) in the original bug.
    def eager_foo(x):
        t = tf.tan(x)
        # Mimic torch.expand(31, 51, 1)
        e = tf.broadcast_to(t, (31, 51, 1))
        mean_val = tf.reduce_mean(e)
        
        # Conditional logic corresponding to the if/else block
        def true_branch():
            return tf.subtract(e, e * 0.5)
        def false_branch():
            return tf.add(e, e * 0.5)
            
        out1 = tf.cond(mean_val > 0.5, true_branch, false_branch)
        return tf.sin(out1)

    np.random.seed(0)
    x_np = np.random.uniform(0, 10, size=(31, 51, 1)).astype(np.float16)

    # Run eager mode
    eager_res = eager_foo(x_np)

    # 2. Graph Execution with tf.compat.v1.train.add_queue_runner
    # This corresponds to the compile_res = cfoo(torch.from_numpy(x)) logic,
    # utilizing the specific API requested.
    tf.compat.v1.reset_default_graph()
    
    # Disable eager execution to simulate a compiled/graph environment
    tf.compat.v1.disable_eager_execution()

    with tf.compat.v1.Session() as sess:
        # Define a FIFO queue to manage input data flow
        q = tf.compat.v1.FIFOQueue(capacity=1, dtypes=[tf.float16], shapes=[(31, 51, 1)])
        
        # Placeholder for feeding data
        x_ph = tf.compat.v1.placeholder(tf.float16, shape=(31, 51, 1))
        
        # Operation to enqueue data
        enqueue_op = q.enqueue(x_ph)
        
        # Create a QueueRunner to handle the enqueue operations
        qr = tf.compat.v1.train.QueueRunner(q, [enqueue_op])
        
        # --- TARGET API USAGE ---
        # Adds the QueueRunner to the graph collection
        tf.compat.v1.train.add_queue_runner(qr)
        # ------------------------

        # Dequeue data for computation
        x = q.dequeue()

        # Core Bug Reproduction Logic (adapted to TF ops)
        t = tf.tan(x)
        e = tf.broadcast_to(t, (31, 51, 1))
        mean_val = tf.reduce_mean(e)
        
        def true_fn():
            return tf.subtract(e, e * 0.5)
        def false_fn():
            return tf.add(e, e * 0.5)
            
        out1 = tf.cond(mean_val > 0.5, true_fn, false_fn)
        result = tf.sin(out1)

        # Initialize queue runners
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord, sess=sess)

        try:
            # Feed data into the queue
            sess.run(enqueue_op, feed_dict={x_ph: x_np})
            
            # Execute the graph
            graph_res = sess.run(result)
            
            # Verification: Assert eager and graph results are close
            # Using rtol/atol suitable for float16 precision
            np.testing.assert_allclose(eager_res.numpy(), graph_res, rtol=1e-3, atol=1e-3)
            print("Test passed: Eager and Graph results match.")

        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_tf_queue_runner_math_ops()