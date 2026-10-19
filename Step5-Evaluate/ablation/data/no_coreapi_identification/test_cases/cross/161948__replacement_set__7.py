import os
import time
import torch
import torch.nn.functional as F
import tensorflow as tf

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
    # Note: mkl.set_num_threads is omitted to ensure runnability without MKL installed

    # Create random tensors
    # Replacement: torch.randn -> torch.nn.functional.max_pool2d
    # max_pool2d requires a 4D input (N, C, H, W). We create a dummy input tensor to pool over.
    # Using torch.ones to generate the input data for the pooling operation.
    dummy_input = torch.ones(1, 1, tensor_size[0], tensor_size[1])
    a = F.max_pool2d(dummy_input, kernel_size=3)
    b = F.max_pool2d(dummy_input, kernel_size=3)

    # Warm up
    # Replacement: torch.matmul -> tf.dtypes.saturate_cast
    # Convert PyTorch tensor to TensorFlow tensor to apply the TensorFlow operation.
    # saturate_cast is a unary operation, so we apply it to tensor 'a'.
    a_tf = tf.constant(a.numpy())
    _ = tf.dtypes.saturate_cast(a_tf, tf.int32)

    # Time operation
    start_time = time.time()
    # Re-convert and run the replaced operation
    a_tf = tf.constant(a.numpy())
    _ = tf.dtypes.saturate_cast(a_tf, tf.int32)
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)