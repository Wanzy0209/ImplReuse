import tensorflow as tf
import numpy as np

# Ensure TF1 graph mode is active, as add_queue_runner is not compatible with eager execution
tf.compat.v1.disable_eager_execution()

def test_queue_runner_with_complex_ops():
    """
    Adapts the PyTorch bug reproduction logic to TensorFlow.
    The original bug involves operations (sum, mean, cumsum) that trigger specific 
    compilation paths (helper functions). Here we verify that these operations 
    can be processed and enqueued using the target API: add_queue_runner.
    """
    
    # Define inputs matching the shapes from the PyTorch bug report
    # PyTorch: (16, 128), (32, 128), (32, 256)
    # We use tf.random.uniform to generate data within the graph so the QueueRunner
    # can execute autonomously without requiring external feed_dict for every step.
    x = tf.random.uniform([16, 128], name='x')
    y = tf.random.uniform([32, 128], name='y')
    z = tf.random.uniform([32, 256], name='z')

    # Define the operations from the bug report
    # x.sum(1), y.mean(1), z.cumsum(1)
    out_sum = tf.reduce_sum(x, axis=1)
    out_mean = tf.reduce_mean(y, axis=1)
    out_cumsum = tf.cumsum(z, axis=1)

    # Setup a FIFOQueue to hold the results
    # Shapes correspond to the output of the operations above
    queue = tf.compat.v1.FIFOQueue(
        capacity=10, 
        dtypes=[tf.float32, tf.float32, tf.float32], 
        shapes=[[128], [128], [32, 256]],
        name='result_queue'
    )
    
    # Operation to enqueue the computed results
    enqueue_op = queue.enqueue([out_sum, out_mean, out_cumsum])
    
    # Create a QueueRunner to manage the enqueueing threads
    # This mimics the "execution" phase of the compiled function
    qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op])
    
    # --- Target API Call ---
    # Add the QueueRunner to the graph collection
    tf.compat.v1.train.add_queue_runner(qr)
    # -----------------------

    with tf.compat.v1.Session() as sess:
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        
        # Start the queue runner threads
        coord = tf.train.Coordinator()
        threads = qr.create_threads(sess, coord=coord, start=True)
        
        try:
            # Attempt to dequeue results to verify the pipeline works
            # We run this a couple of times to ensure the queue is populated by the runner
            for i in range(2):
                r_sum, r_mean, r_cumsum = sess.run(queue.dequeue())
                
                # Assertions to verify the output shapes match expectations
                assert r_sum.shape == (128,), f"Expected sum shape (128,), got {r_sum.shape}"
                assert r_mean.shape == (128,), f"Expected mean shape (128,), got {r_mean.shape}"
                assert r_cumsum.shape == (32, 256), f"Expected cumsum shape (32, 256), got {r_cumsum.shape}"
                
            print("Test passed: Operations executed and queued successfully via add_queue_runner.")
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            raise
        finally:
            # Stop the threads
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_queue_runner_with_complex_ops()