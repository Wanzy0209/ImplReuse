```python
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
# Conversion: mkl is not directly exposed in TensorFlow; threading is managed via tf.config

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
    # Note: In TensorFlow, thread pools are initialized at startup. 
    # Changing configuration dynamically inside a loop may not always resize existing pools.
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    # Conversion: Set inter-op threads to 1 to isolate intra-op (BLAS) performance
    tf.config.threading.set_inter_op_parallelism_threads(1)

    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    # Conversion: mkl.set_num_threads is handled by tf.config or environment variables in TF

    # Create random tensors
    # Conversion: torch.randn -> tf.random.normal
    a = tf.random.normal(tensor_size, dtype=tf.float32)
    b = tf.random.normal(tensor_size, dtype=tf.float32)

    # Warm up
    # Conversion: torch.matmul -> tf.matmul
    _ = tf.matmul(a, b)

    # Time matrix multiplication
    start_time = time.time()
    _ = tf.matmul(a, b)
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)
```