import os
import time
import torch
import tensorflow as tf

# Optional import for MKL, handled gracefully if missing to ensure runnability
try:
    import mkl
except ImportError:
    mkl = None

# List of threads to test
threads_list = [1, 2, 4, 8, 16, 32, 48]

# Size of the large tensors
tensor_size = (10000, 10000)

# Store results
times = []

# Benchmark for each thread count
for threads in threads_list:
    # Set environment variables and thread count
    torch.set_num_threads(int(threads))
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    if mkl:
        mkl.set_num_threads(int(threads))

    # Replacement: torch.randn -> torch.var_mean
    # torch.var_mean requires an input tensor. We create a dummy tensor to compute variance and mean.
    dummy_input = torch.ones(tensor_size)
    a = torch.var_mean(dummy_input)
    b = torch.var_mean(dummy_input)

    # Replacement: torch.matmul -> tf.keras.backend.switch
    # tf.keras.backend.switch requires a condition and two expressions.
    # We use dummy TensorFlow constants to ensure the call is valid and runnable.
    
    # Warm up
    _ = tf.keras.backend.switch(tf.constant(True), tf.constant(1.0), tf.constant(0.0))

    # Time operation
    start_time = time.time()
    _ = tf.keras.backend.switch(tf.constant(True), tf.constant(1.0), tf.constant(0.0))
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)