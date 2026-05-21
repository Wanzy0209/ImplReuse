import tensorflow as tf
import gc
import sys
import tracemalloc

def test_queue_runner_memory_leak():
    """
    Adapted test case for tf.compat.v1.train.add_queue_runner.
    
    Original Bug Context:
    The PyTorch issue (165407) describes a memory leak where the number of 
    tensors increases every step during a compiled training loop involving 
    flash_attn_varlen_func.
    
    Adaptation Logic:
    This test verifies the behavior of the TensorFlow QueueRunner API in a 
    loop. While TensorFlow v1 uses a static graph (where tensor definitions 
    don't change at runtime), we monitor the process memory and object counts 
    to ensure that the queue runners and background threads managed by 
    add_queue_runner do not cause unbounded memory growth during execution.
    """
    
    # Ensure we are in Graph Mode (required for compat.v1 APIs)
    tf.compat.v1.disable_eager_execution()

    # 1. Setup the Graph and Queue
    with tf.compat.v1.Graph().as_default():
        # Create a simple FIFO queue
        queue = tf.compat.v1.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=[()])
        
        # Define an enqueue operation
        enqueue_op = queue.enqueue([1.0])
        
        # Create a QueueRunner that will run the enqueue op in a background thread
        # We create 2 threads to mimic some parallelism
        qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op] * 2)
        
        # --- API Under Test ---
        # Add the QueueRunner to the default collection
        tf.compat.v1.train.add_queue_runner(qr)
        
        # Define a dequeue operation to consume data in the loop
        dequeue_op = queue.dequeue()

        # 2. Execute the Session
        with tf.compat.v1.Session() as sess:
            # Initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())
            
            # Start the queue runners
            coord = tf.compat.v1.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

            # Start tracing memory
            tracemalloc.start()
            
            try:
                # 3. Run a loop to simulate training steps
                for step in range(300):
                    # Perform a step (dequeue data)
                    _ = sess.run(dequeue_op)
                    
                    # 4. Monitor Memory and Object Counts
                    # We check periodically to match the style of the original bug report
                    if step > 0 and step % 50 == 0:
                        gc.collect()
                        
                        # Get memory usage
                        current, peak = tracemalloc.get_traced_memory()
                        
                        # Count TensorFlow Tensor objects in memory
                        # Note: In TF v1 graph mode, tensors are graph nodes and usually static,
                        # but we check for any unexpected accumulation of wrapper objects.
                        num_tensors = sum(1 for obj in gc.get_objects() if isinstance(obj, tf.Tensor))
                        
                        print(f"Step {step:4d} | Alloc: {current / 1e9:.3f}GB | Peak: {peak / 1e9:.3f}GB | Tensors: {num_tensors}")
                        
                        # Assertion to detect leaks (adjust threshold as necessary for environment)
                        # In a healthy TF v1 loop, object counts should remain relatively stable.
                        # If this grows indefinitely (like the PyTorch bug), it indicates a leak.
                        if step > 100 and num_tensors > 5000:
                            raise AssertionError(f"Potential memory leak detected: Tensor count {num_tensors} exceeds threshold")

            finally:
                # Cleanup: Stop threads and close session
                coord.request_stop()
                coord.join(threads)
                tracemalloc.stop()

if __name__ == "__main__":
    test_queue_runner_memory_leak()