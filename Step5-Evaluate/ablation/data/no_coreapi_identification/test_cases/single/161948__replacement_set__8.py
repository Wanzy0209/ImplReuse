import os
import time
import torch
import torch.nn.functional as F
import tensorflow as tf

# List of threads to test
threads_list = [1, 2, 4]

# Size of the large tensors
# Original shape (10000, 10000) is 2D. 
# torch.nn.functional.max_pool3d requires 5D input (Batch, Channels, Depth, Height, Width).
# We adapt the shape to be compatible with the replacement API.
tensor_shape = (1, 1, 10, 100, 100)

# Store results
times = []

# Benchmark for each thread count
for threads in threads_list:
    # Set environment variables and thread count
    torch.set_num_threads(int(threads))
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)

    # Create random tensors
    # Replacement: torch.randn -> torch.nn.functional.max_pool3d
    # Since max_pool3d is an operation and not a generator, we create a dummy input using torch.ones
    # to feed into the pooling operation, as torch.randn is being replaced.
    dummy_input = torch.ones(tensor_shape)
    a = F.max_pool3d(dummy_input, kernel_size=2)
    b = F.max_pool3d(dummy_input, kernel_size=2)

    # Warm up
    # Replacement: torch.matmul -> tf.keras.backend.resize_images
    # We must convert the PyTorch tensor 'a' to a TensorFlow tensor to use the TF API.
    # We use data_format='channels_first' to match PyTorch's (N, C, D, H, W) convention.
    tf_a = tf.convert_to_tensor(a.numpy())
    _ = tf.keras.backend.resize_images(tf_a, height_factor=2, width_factor=2, data_format='channels_first')

    # Time operation
    start_time = time.time()
    tf_a = tf.convert_to_tensor(a.numpy())
    _ = tf.keras.backend.resize_images(tf_a, height_factor=2, width_factor=2, data_format='channels_first')
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)

# Assertion to verify the test ran through all thread counts
assert len(times) == len(threads_list)