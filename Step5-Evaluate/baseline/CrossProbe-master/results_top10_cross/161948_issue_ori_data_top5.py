```python
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
# mkl is not directly exposed in TensorFlow, threading is managed via tf.config

# List of threads to test
threads_list = [1, 2, 4, 8, 16, 32, 48]

# Size of the large tensors
tensor_size = (10000, 10000)

# Store results
times = []

# Benchmark for each thread count
for threads in threads_list:
    # Set environment variables and thread count
    # torch.set_num_threads -> tf.config.threading.set_intra_op_parallelism_threads
    tf.config.threading.set_intra_op_parallelism_threads(int(threads))
    # Also setting inter-op threads to match source behavior of controlling all parallelism
    tf.config.threading.set_inter_op_parallelism_threads(int(threads))
    
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    # mkl.set_num_threads is handled by tf.config and environment variables in TF

    # Create random tensors
    # torch.randn -> tf.random.normal
    a = tf.random.normal(tensor_size)
    b = tf.random.normal(tensor_size)

    # Warm up
    # torch.matmul -> tf.linalg.matmul
    _ = tf.linalg.matmul(a, b)

    # Time matrix multiplication
    start_time = time.time()
    _ = tf.linalg.matmul(a, b)
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)
```