import tensorflow as tf
import numpy as np

def test_tf_queue_runner_divergence():
    """
    Adapted test case for tf.compat.v1.train.add_queue_runner.
    Original PyTorch issue: Divergence between eager and compiled mode 
    involving scalar operations and division.
    
    This test verifies that the logic behaves consistently when run eagerly
    versus when embedded in a TensorFlow graph managed by QueueRunners.
    """
    
    # Setup input similar to the PyTorch fuzzer
    # arg_0 is a random int32 scalar
    np.random.seed(19989)
    arg_0_val = np.int32(np.random.randn())
    sentinel_val = 1.0

    # --- 1. Eager Execution ---
    # In TF 2.x, eager is enabled by default.
    print("Running Eager Execution...")
    
    # Replicate the logic
    var_node_2 = tf.constant(-6, dtype=tf.int64)
    var_node_3 = tf.cast(arg_0_val, dtype=tf.int32)
    var_node_1 = var_node_2 * var_node_3
    var_node_5 = tf.fill((), 1, dtype=tf.int64)
    var_node_4 = var_node_5 # In eager, this is a Tensor
    var_node_0 = var_node_1 / var_node_4
    result_eager = var_node_0 * sentinel_val
    
    # Convert to numpy for comparison
    result_eager_val = result_eager.numpy()
    print(f" Eager Result: {result_eager_val}")

    # --- 2. Graph Execution with QueueRunner ---
    # This mimics the 'compiled' mode in PyTorch where operations are graph-based.
    print("Running Graph Execution with QueueRunner...")
    
    # Reset graph to ensure clean state
    tf.compat.v1.reset_default_graph()
    
    # Disable eager execution to enter graph mode (required for compat.v1.QueueRunner)
    tf.compat.v1.disable_eager_execution()

    with tf.compat.v1.Session() as sess:
        # Define placeholders for inputs
        arg_0_ph = tf.compat.v1.placeholder(dtype=tf.int32, shape=())
        sentinel_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=())

        # Replicate the logic in the graph
        var_node_2_g = tf.constant(-6, dtype=tf.int64)
        var_node_3_g = tf.cast(arg_0_ph, dtype=tf.int32)
        var_node_1_g = var_node_2_g * var_node_3_g
        var_node_5_g = tf.fill((), 1, dtype=tf.int64)
        var_node_4_g = var_node_5_g
        var_node_0_g = var_node_1_g / var_node_4_g
        result_g = var_node_0_g * sentinel_ph

        # To utilize tf.compat.v1.train.add_queue_runner, we need a Queue.
        # We create a FIFOQueue to enqueue the result, simulating a data pipeline step.
        queue = tf.compat.v1.queue.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[()])
        
        # Enqueue the computed result
        enqueue_op = queue.enqueue(result_g)
        
        # Create a QueueRunner
        # The QueueRunner will manage the enqueue threads.
        qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op])
        
        # --- API Under Test: tf.compat.v1.train.add_queue_runner ---
        # This adds the QueueRunner to the graph collection.
        tf.compat.v1.train.add_queue_runner(qr)
        # ----------------------------------------------------------------

        # Coordinator for managing threads
        coord = tf.compat.v1.train.Coordinator()
        
        # Start the queue runners
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        try:
            # Run the enqueue operation with the provided inputs
            # This triggers the computation graph logic.
            sess.run(enqueue_op, feed_dict={arg_0_ph: arg_0_val, sentinel_ph: sentinel_val})
            
            # Dequeue the result to verify
            result_graph_val = sess.run(queue.dequeue())
            print(f" Graph Result: {result_graph_val}")

            # Verify consistency between Eager and Graph modes
            # Note: We use np.isclose because floating point arithmetic might differ slightly
            # or type casting might behave differently between eager/graph implementations.
            if np.isclose(result_eager_val, result_graph_val):
                print(" Success: Eager and Graph results match.")
            else:
                print(f" Divergence detected: Eager={result_eager_val}, Graph={result_graph_val}")
                raise AssertionError("Divergence between eager and graph execution")

        finally:
            # Stop threads
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_tf_queue_runner_divergence()