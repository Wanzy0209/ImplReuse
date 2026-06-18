import torch
import threading
import time

def eager_all(x: torch.Tensor):
    return torch.all(x > 0)

@torch.compile
def compiled_all(x: torch.Tensor):
    return torch.all(x > 0)

def check_gil_released(func, *args, **kwargs):
    """
    Helper to check if the GIL is released during the execution of func.
    Returns a tuple (result, gil_released_bool).
    """
    gil_released = [False]

    def worker():
        # This thread will only run if the main thread releases the GIL
        gil_released[0] = True

    t = threading.Thread(target=worker)
    t.start()

    # Call the function under test
    result = func(*args, **kwargs)

    # Wait a brief moment for the thread to potentially run
    t.join(timeout=0.1)

    return result, gil_released[0]

def main():
    # Create a large tensor to ensure the operation takes enough time
    # for the GIL check to be meaningful
    x = torch.randn(4096, 4096, device='cuda')

    print("Testing eager torch.all...")
    _, released = check_gil_released(eager_all, x)
    print(f"Eager torch.all released GIL: {released}")
    assert released, "Expected eager torch.all to release GIL"

    print("Testing compiled torch.all...")
    # Warm up compilation
    compiled_all(x)
    
    _, released = check_gil_released(compiled_all, x)
    print(f"Compiled torch.all released GIL: {released}")
    
    # The bug report suggests that torch.compile kernels hold the GIL.
    # This assertion checks if the similar API (torch.all) exhibits the same behavior.
    # If this fails, it confirms the bug exists for this API as well.
    if not released:
        print("WARNING: Compiled torch.all held the GIL (Bug reproduced).")
    else:
        print("SUCCESS: Compiled torch.all released the GIL.")

if __name__ == "__main__":
    main()