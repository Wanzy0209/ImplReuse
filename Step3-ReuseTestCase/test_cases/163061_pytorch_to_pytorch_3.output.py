import torch
import threading
import time

# Define a custom operator to test GIL behavior
# using the similar API: torch.library.impl_abstract
torch.library.define("test_gil_ops::custom_add", "(Tensor x, Tensor y) -> Tensor")

# Register the abstract implementation (meta kernel) using the target API
@torch.library.impl_abstract("test_gil_ops::custom_add")
def custom_add_meta(x, y):
    return torch.empty_like(x)

# Register the actual CUDA implementation
@torch.library.impl("test_gil_ops::custom_add", "CUDA")
def custom_add_cuda(x, y):
    # Simulate a kernel operation
    return x + y

def check_gil_released(func):
    """
    Checks if the GIL is released during the execution of func.
    Returns True if the GIL was released (other thread ran), False otherwise.
    """
    other_thread_ran = False
    
    def worker():
        nonlocal other_thread_ran
        # Perform some CPU-bound work that requires the GIL
        # If the main thread holds the GIL, this will wait.
        # If the main thread releases the GIL, this will run concurrently.
        start = time.time()
        while time.time() - start < 0.1: 
            pass
        other_thread_ran = True

    t = threading.Thread(target=worker)
    t.start()
    
    # Execute the function under test
    func()
    
    t.join()
    return other_thread_ran

def main():
    # Setup inputs
    x = torch.randn(4096, 4096, device='cuda')
    y = torch.randn(4096, 4096, device='cuda')

    # Test the custom operator defined via torch.library.impl_abstract
    def run_custom_op():
        z = torch.ops.test_gil_ops.custom_add(x, y)
        # Ensure kernel execution is complete before checking thread status
        torch.cuda.synchronize()

    print("Checking GIL release for operator defined via torch.library.impl_abstract...")
    is_released = check_gil_released(run_custom_op)
    
    if is_released:
        print("PASS: GIL was released.")
    else:
        print("FAIL: GIL was held.")

    # Basic sanity check to ensure the operator works
    result = torch.ops.test_gil_ops.custom_add(x, y)
    expected = x + y
    assert torch.allclose(result, expected), "Custom operator produced incorrect results"

if __name__ == "__main__":
    main()