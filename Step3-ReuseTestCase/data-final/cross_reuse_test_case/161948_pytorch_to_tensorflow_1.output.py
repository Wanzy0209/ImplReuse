import torch
import os
import time
import tensorflow as tf

# List of threads to test
threads_list = [1, 2, 4, 8, 16, 32, 48]

# Size of the large tensors
tensor_size = (10000, 10000)

# Store results
times = []

# Benchmark for each thread count
for threads in threads_list:
    # Set TensorFlow threading configuration (equivalent to torch.set_num_threads)
    # intra_op_parallelism_threads controls parallelism for individual operations (like gather)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    # inter_op_parallelism_threads controls parallelism between distinct operations
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    
    # Set environment variables (preserving original logic)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)

    # Create random tensors
    # tf.gather requires params (data) and indices
    params = tf.random.normal(tensor_size)
    # Create indices to gather a significant portion of the data to ensure workload
    # Here we gather every other row
    indices = tf.range(start=0, limit=tensor_size[0], delta=2)

    # Warm up
    _ = tf.gather(params, indices, axis=0)

    # Time the gather operation
    start_time = time.time()
    _ = tf.gather(params, indices, axis=0)
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)