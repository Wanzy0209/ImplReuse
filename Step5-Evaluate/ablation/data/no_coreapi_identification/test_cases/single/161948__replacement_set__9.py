import os
import time
import torch
import torch.nn.functional as F

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

    # Create inputs for the loss function
    # Note: torch.randn is replaced by torch.nn.functional.hinge_embedding_loss
    # We use torch.ones to generate the necessary inputs for the loss function.
    input_tensor = torch.ones(tensor_size)
    target_tensor = torch.ones(tensor_size)
    
    # Replacement: torch.randn -> torch.nn.functional.hinge_embedding_loss
    # We use reduction='none' to maintain tensor dimensions for the subsequent operation
    a = F.hinge_embedding_loss(input_tensor, target_tensor, reduction='none')

    # Warm up
    # Replacement: torch.matmul -> torch.repeat_interleave
    _ = torch.repeat_interleave(a, repeats=2)

    # Time operation
    start_time = time.time()
    # Replacement: torch.matmul -> torch.repeat_interleave
    _ = torch.repeat_interleave(a, repeats=2)
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)

# Basic assertion to verify the test ran for all thread counts
assert len(times) == len(threads_list)