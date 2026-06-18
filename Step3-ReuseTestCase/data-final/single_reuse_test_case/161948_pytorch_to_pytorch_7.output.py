import os
import time
import torch
import mkl

# List of threads to test
threads_list = [1, 2, 4, 8, 16, 32, 48]

# Dimensions for the indices
n = 10000
m = 10000
offset = 0

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

    # Warm up
    _ = torch.triu_indices(n, m, offset=offset)

    # Time the operation
    start_time = time.time()
    indices = torch.triu_indices(n, m, offset=offset)
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)

    # Verify correctness of the API output
    # Check shape (should be 2 rows)
    assert indices.shape[0] == 2, f"Expected 2 rows in indices, got {indices.shape[0]}"
    
    # Check bounds
    assert indices[0].max() < n, f"Row index out of bounds"
    assert indices[1].max() < m, f"Column index out of bounds"
    
    # Check upper triangular property: row <= col + offset
    assert torch.all(indices[0] <= indices[1] + offset), "Indices do not satisfy upper triangular condition"

print("Test completed successfully.")