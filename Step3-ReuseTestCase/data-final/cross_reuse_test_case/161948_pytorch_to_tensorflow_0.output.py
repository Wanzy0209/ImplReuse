import os
import time
import tensorflow as tf

# List of threads to test
threads_list = [1, 2, 4, 8, 16, 32, 48]

# Size of the large tensors
# For gather, we need params and indices. We define shapes large enough to measure performance.
params_shape = (10000, 100)
indices_shape = (10000, 100)

# Store results
times = []

# Benchmark for each thread count
for threads in threads_list:
    # Set TensorFlow thread configuration
    # Note: In TensorFlow, thread configuration should ideally happen at program start.
    # We set it here to mirror the structure of the original PyTorch test case.
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)

    # Create random tensors
    # Using tf.random.normal to generate data
    params = tf.random.normal(params_shape)
    indices = tf.random.uniform(indices_shape, minval=0, maxval=params_shape[0], dtype=tf.int32)

    # Warm up
    _ = tf.compat.v1.gather(params, indices)

    # Time the gather operation
    start_time = time.time()
    result = tf.compat.v1.gather(params, indices)
    
    # Ensure the operation is executed (especially important in TF2 eager mode)
    _ = result.numpy()
    
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)

# Basic assertion to verify the API executed correctly and produced the expected shape
assert result.shape == indices_shape