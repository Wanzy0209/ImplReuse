import os
import time
import torch

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

    # Create random tensor
    a = torch.randn(tensor_size)

    # Warm up
    _ = torch.count_nonzero(a)

    # Time count_nonzero operation
    start_time = time.time()
    result = torch.count_nonzero(a)
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)

    # Verify the result is valid
    assert result.item() >= 0, "Count of non-zero elements must be non-negative"
    assert result.item() <= tensor_size[0] * tensor_size[1], "Count cannot exceed total elements"