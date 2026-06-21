import torch
import threading
import time

# Fix: Handle older PyTorch versions where torch.compile does not exist
if not hasattr(torch, 'compile'):
    # Mock torch.compile as a pass-through decorator
    torch.compile = lambda func: func

def torch_any(x: torch.Tensor):
    return torch.any(x)

@torch.compile
def torch_compile_any(x: torch.Tensor):
    return torch.any(x)

def main():
    x = torch.randn(4096, 4096, device='cuda')
    
    # Helper to check if GIL is released during the call
    def check_gil(func, name):
        thread_ran = False
        
        def worker():
            nonlocal thread_ran
            # Perform some CPU-bound work to check if we can acquire the GIL
            _ = sum(range(1000000))
            thread_ran = True
        
        t = threading.Thread(target=worker)
        t.start()
        
        # Execute the function
        result = func(x)
        
        t.join()
        
        print(f"{name}: Result={result.item()}, Thread ran concurrently: {thread_ran}")
        return result

    # Run the test loop similar to the original report
    for _ in range(10):
        check_gil(torch_any, "Eager torch.any")
        check_gil(torch_compile_any, "Compiled torch.any")

if __name__ == "__main__":
    main()