import torch
import os

# List of threads to test
threads_list = [1, 2, 4]

# Size of the tensors
tensor_size = (1000, 1000)

# Test for each thread count
for threads in threads_list:
    # Set environment variables and thread count
    torch.set_num_threads(int(threads))
    os.environ['OMP_NUM_THREADS'] = str(threads)

    # Create random tensors
    a = torch.randn(tensor_size)
    b = a.clone() # Create an identical tensor for equality check
    c = torch.randn(tensor_size) # Create a different tensor for inequality check

    # Test torch.eq (Similar API)
    # Verify that torch.eq correctly identifies equal elements
    result_eq = torch.eq(a, b)
    
    # Verify that torch.eq correctly identifies unequal elements
    result_neq = torch.eq(a, c)

    # Assertions
    assert result_eq.dtype == torch.bool, "Result dtype should be boolean"
    assert result_eq.all(), "torch.eq failed: All elements should be equal for cloned tensors"
    assert not result_neq.all(), "torch.eq failed: Not all elements should be equal for random tensors"

    print(f"Test passed for torch.eq with {threads} threads.")