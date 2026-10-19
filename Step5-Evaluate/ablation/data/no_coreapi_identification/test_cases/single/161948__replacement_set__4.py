import os
import time
import torch
import torch.nn.functional as F
import tensorflow as tf
import matplotlib.pyplot as plt
import mkl

# List of threads to test
threads_list = [1, 2]

# Size of the large tensors
tensor_size = (100, 100)

# Store results
times = []

# Benchmark for each thread count
for threads in threads_list:
    # Set environment variables and thread count
    torch.set_num_threads(int(threads))
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    mkl.set_num_threads(int(threads))

    # Create random tensors
    # Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool2d
    # We need a dummy input to perform the pooling operation.
    dummy_input = torch.randn(1, 1, 200, 200)
    
    # Apply replacement API to generate 'a'
    a = F.adaptive_max_pool2d(dummy_input, output_size=tensor_size)
    # Reshape to 2D to match original tensor_size structure
    a = a.view(tensor_size)

    # Apply replacement API to generate 'b'
    b = F.adaptive_max_pool2d(dummy_input, output_size=tensor_size)
    b = b.view(tensor_size)

    # Warm up
    # Replacement: torch.matmul -> tf.signal.overlap_and_add
    # Convert torch tensor to tensorflow tensor
    a_tf = tf.convert_to_tensor(a.numpy())
    # overlap_and_add requires a frame_step. 
    # Input shape is (100, 100), treated as (frames, frame_length).
    _ = tf.signal.overlap_and_add(a_tf, frame_step=50)

    # Time operation
    start_time = time.time()
    # Replacement: torch.matmul -> tf.signal.overlap_and_add
    result = tf.signal.overlap_and_add(a_tf, frame_step=50)
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)

    # Assertion to verify the operation executed and produced expected shape
    # Expected shape for overlap_and_add: (frames - 1 + frame_length)
    # (100 - 1 + 50) = 149
    assert result.shape == (149,), f"Expected shape (149,), got {result.shape}"