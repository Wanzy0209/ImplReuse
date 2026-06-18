import os
import time
import torch
import matplotlib.pyplot as plt
import mkl

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
    mkl.set_num_threads(int(threads))

    # Create random tensors
    a = torch.randn(tensor_size)
    b = torch.randn(tensor_size)

    # Warm up
    _ = torch.matmul(a, b)

    # Time matrix multiplication
    start_time = time.time()
    _ = torch.matmul(a, b)
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)