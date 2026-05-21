import torch
import gc

def test_torch_all_compile_memory_leak():
    """
    Test case to verify if torch.all causes a memory leak when used with torch.compile.
    This is adapted from Issue #165407 which reported a memory leak involving 
    flash_attn_varlen_func under torch.compile.
    """
    
    # Define a function using torch.all, the similar API identified
    def func(x):
        # torch.all is used here to check for similar memory retention issues
        return torch.all(x > 0)

    # Compile the function using torch.compile (the context of the original bug)
    compiled_func = torch.compile(func)

    tensor_counts = []
    
    print("Starting memory leak test for torch.all with torch.compile...")

    for i in range(10):
        # Create input tensor
        x = torch.randn(128, 128)
        
        # Execute the compiled function
        res = compiled_func(x)
        
        # Explicitly delete references to help garbage collection
        del x, res
        gc.collect()
        
        # Count the number of live tensors to detect leaks
        # This mimics the "Tensors: X" output in the original bug report
        live_tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]
        count = len(live_tensors)
        tensor_counts.append(count)
        
        print(f"Step {i} | Tensors: {count}")

    # Analyze the trend
    # In the original bug, tensor counts increased steadily (e.g., 2266 -> 3514).
    # We check if the final average is significantly higher than the initial average.
    initial_avg = sum(tensor_counts[:3]) / 3
    final_avg = sum(tensor_counts[-3:]) / 3
    
    # We allow a small buffer for legitimate caching by the compiler/allocator,
    # but a 50% increase indicates a leak similar to the original report.
    if final_avg > initial_avg * 1.5:
        raise AssertionError(
            f"Memory leak detected! "
            f"Average tensor count grew from {initial_avg:.2f} to {final_avg:.2f}."
        )
    
    print("Test passed. No significant memory leak detected.")

if __name__ == "__main__":
    test_torch_all_compile_memory_leak()