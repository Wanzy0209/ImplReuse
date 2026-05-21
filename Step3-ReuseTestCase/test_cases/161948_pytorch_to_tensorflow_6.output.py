import torch
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt

# List of threads to test
threads_list = [1, 2, 4, 8, 16, 32, 48]

# Size of the large tensors
# Note: SVD is computationally more expensive than MatMul. 
# We keep the size consistent with the original test case logic.
tensor_size = (10000, 10000)

# Store results
times = []

# Benchmark for each thread count
for threads in threads_list:
    # Set TensorFlow threading configuration
    # This is the TensorFlow equivalent to torch.set_num_threads
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    
    # Set environment variables (similar to original logic)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)

    # Create random tensors
    a = tf.random.normal(tensor_size)

    # Warm up
    _ = tf.keras.ops.svd(a)

    # Time SVD operation (The target API)
    start_time = time.time()
    _ = tf.keras.ops.svd(a)
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)

# Optional: Plotting results to visualize scaling
plt.figure(figsize=(10, 6))
plt.plot(threads_list, times, marker='o')
plt.title('TensorFlow SVD Performance vs Thread Count')
plt.xlabel('Number of Threads')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.show()