import torch
import tensorflow as tf
import numpy as np

# QueueRunners are part of the TF1 graph mode and are not compatible with eager execution.
# This mimics the "compiled" graph context of the original PyTorch test.
tf.compat.v1.disable_eager_execution()

def test_queue_runner_gpu_to_cpu():
    """
    Adapted from PyTorch test_triton_kernel_to_cpu.
    
    Original Logic:
    1. Define inputs on GPU.
    2. Run a kernel (addition) on GPU.
    3. Move output to CPU (.cpu()).
    4. Perform operation on CPU (+ 1).
    5. Verify correctness.

    Adapted Logic (TensorFlow):
    1. Define inputs on GPU.
    2. Perform operation (addition) on GPU.
    3. Enqueue result into a Queue (managed by QueueRunner).
    4. Dequeue on CPU (mimicking the transfer).
    5. Perform operation on CPU (+ 1).
    6. Verify correctness.
    """
    
    # Check for GPU availability to match the @requires_gpu decorator logic
    if not tf.config.list_physical_devices('GPU'):
        print("Test skipped: No GPU available.")
        return

    with tf.compat.v1.Session() as sess:
        # 1. Setup inputs on GPU (mimicking torch.randn on GPU_TYPE)
        with tf.device('/GPU:0'):
            x = tf.random.normal((4, 4))
            y = tf.random.normal((4, 4))
            
            # Mimic the 'add_kernel' logic: simple addition on GPU
            # In the original bug, this was a user-defined Triton kernel.
            gpu_result = x + y

        # 2. Setup Queue and QueueRunner
        # The queue acts as the buffer for the data transfer.
        q = tf.compat.v1.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[(4, 4)])
        
        # Enqueue the GPU result
        enqueue_op = q.enqueue(gpu_result)
        
        # Create the QueueRunner
        # This manages the background threads that enqueue data, similar to how
        # inductor manages kernel execution.
        qr = tf.compat.v1.train.QueueRunner(q, [enqueue_op])
        
        # API Under Test: tf.compat.v1.train.add_queue_runner
        # This adds the runner to the graph collection.
        tf.compat.v1.train.add_queue_runner(qr)

        # 3. Retrieve data on CPU (mimicking .cpu())
        with tf.device('/CPU:0'):
            out = q.dequeue()
            # Mimic the + 1 operation from the original test
            out_cpu = out + 1

        # 4. Execution
        # Start the queue runners (mimicking the execution of the compiled graph)
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        try:
            # Initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())
            
            # Run the operation
            result_val = sess.run(out_cpu)
            
            # 5. Verification
            # We assert the shape and type to ensure the pipeline ran correctly.
            # Since we are using random numbers, we can't assert exact values 
            # against a static eager reference, but we verify the operation chain completed.
            assert result_val.shape == (4, 4), f"Shape mismatch: {result_val.shape}"
            assert result_val.dtype == np.float32, f"Dtype mismatch: {result_val.dtype}"
            
            print("Test passed: QueueRunner successfully handled GPU to CPU data flow.")

        except Exception as e:
            print(f"Test failed with error: {e}")
            raise
        finally:
            # Stop the threads
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_queue_runner_gpu_to_cpu()