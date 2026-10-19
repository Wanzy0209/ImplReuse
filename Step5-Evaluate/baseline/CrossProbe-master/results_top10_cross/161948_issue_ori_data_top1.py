```python
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt

# List of threads to test
threads_list = [1, 2, 4, 8, 16, 32, 48]

# Size of the large tensors
tensor_size = (10000, 10000)

# Store results
times = []

# Benchmark for each thread count
for threads in threads_list:
    # Set environment variables and thread count
    # Conversion: torch.set_num_threads -> tf.config.threading.set_intra_op_parallelism_threads
    tf.config.threading.set_intra_op_parallelism_threads(int(threads))
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    # Note: mkl.set_num_threads is not available in TensorFlow; threading is managed by tf.config

    # Create random tensors
    # Conversion: torch.randn -> tf.random.normal
    a = tf.random.normal(tensor_size)
    b = tf.random.normal(tensor_size)

    # Warm up
    # Conversion: torch.matmul -> tf.linalg.matmul
    _ = tf.linalg.matmul(a, b)

    # Time matrix multiplication
    start_time = time.time()
    _ = tf.linalg.matmul(a, b)
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)
```