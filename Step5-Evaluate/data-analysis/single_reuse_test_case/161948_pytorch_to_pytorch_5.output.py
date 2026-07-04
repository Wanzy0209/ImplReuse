import torch
import os

# List of threads to test (from original bug report)
threads_list = [1, 2, 4]

# Size of the large tensors (from original bug report)
tensor_size = (10000, 10000)

print("Testing torch.reshape with different thread counts...")

for threads in threads_list:
    # Set environment variables and thread count (context from original bug)
    torch.set_num_threads(int(threads))
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)

    # Create random tensor
    a = torch.randn(tensor_size)

    # --- Adapted Call Site ---
    # Original API: torch.set_num_threads (Configuration)
    # Similar API: torch.reshape (Operation)
    # We verify the similar API here by reshaping the large tensor.
    
    # Reshape to a flat vector
    new_shape = (tensor_size[0] * tensor_size[1],)
    reshaped_a = torch.reshape(a, new_shape)

    # Verify shape
    assert reshaped_a.shape == new_shape, f"Failed: Expected shape {new_shape}, got {reshaped_a.shape}"

    # Verify data integrity (check first and last elements to ensure data is preserved)
    assert reshaped_a[0] == a[0, 0], "Failed: First element mismatch"
    assert reshaped_a[-1] == a[-1, -1], "Failed: Last element mismatch"

    print(f"Threads: {threads}, Reshape test passed.")