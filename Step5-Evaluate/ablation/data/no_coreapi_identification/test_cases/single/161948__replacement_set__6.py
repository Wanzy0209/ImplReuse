import os
import time
import torch
import torch.nn.functional as F
import tensorflow as tf

# List of threads to test
threads_list = [1, 2, 4]

# Size of the large tensors
# Adjusted to (100, 100) to ensure the test remains minimal and runnable 
# given the computational cost of LSTMCell compared to matmul.
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

    # --- Replacement for torch.randn ---
    # Original: a = torch.randn(tensor_size)
    # We use torch.ones to create input data for max_pool1d, then apply the pool.
    # max_pool1d expects (Channels, Length) or (Batch, Channels, Length).
    # Here we use (Channels, Length) matching the 2D tensor_size.
    dummy_input = torch.ones(tensor_size)
    a = F.max_pool1d(dummy_input, kernel_size=2)
    b = F.max_pool1d(dummy_input, kernel_size=2)

    # Convert PyTorch tensors to TensorFlow tensors for the LSTMCell
    # a and b shapes are now (100, 50)
    a_tf = tf.constant(a.numpy())
    b_tf = tf.constant(b.numpy())

    # --- Replacement for torch.matmul ---
    # Original: _ = torch.matmul(a, b)
    # We use tf.keras.layers.LSTMCell.
    # LSTMCell expects input shape (batch, features) and state shape (batch, units).
    # We treat the first dimension of a as batch, and second as features.
    # We set units to match the feature dimension of a (50).
    lstm_cell = tf.keras.layers.LSTMCell(units=50)

    # Warm up
    # LSTMCell call signature: cell(input, states)
    # We use b_tf as the initial state (h and c).
    _ = lstm_cell(a_tf, [b_tf, b_tf])

    # Time operation
    start_time = time.time()
    _ = lstm_cell(a_tf, [b_tf, b_tf])
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)

# Basic assertion to verify the loop completed
assert len(times) == len(threads_list)