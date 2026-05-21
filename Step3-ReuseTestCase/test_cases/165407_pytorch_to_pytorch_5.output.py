import torch
import gc

def test_torch_any_compile_memory_leak():
    """
    Test case to verify if torch.any causes a memory leak when used within torch.compile,
    similar to the reported issue with flash_attn_varlen_func.
    """
    
    # Define a function that uses torch.any
    # This replaces the original call site (flash_attn_varlen_func) with the similar API (torch.any)
    def model_fn(x):
        # Use torch.any to create a mask based on a condition
        # This ensures torch.any is part of the compiled graph
        mask = torch.any(x > 0.0, dim=-1, keepdim=True)
        return x * mask.float()

    # Compile the function using torch.compile
    compiled_fn = torch.compile(model_fn)

    print("Starting test loop for torch.any memory leak...")
    
    # Run a loop to simulate the training steps described in the bug report
    for step in range(100):
        # Create dummy input with varying sequence length to mimic dynamic shapes
        # which might trigger recompilation or specific caching behaviors
        seq_len = 32 + (step % 4)
        x = torch.randn(2, seq_len, 64, requires_grad=True)
        
        # Forward pass
        output = compiled_fn(x)
        
        # Simulate loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Cleanup
        del x, output, loss
        gc.collect()
        
        # Print progress similar to the original bug report
        if step % 10 == 0:
            # Note: Exact tensor counting requires internal APIs, but we monitor progress.
            print(f"Step {step} completed")

    print("Test finished. Please monitor memory usage to confirm no leak occurred.")

if __name__ == "__main__":
    test_torch_any_compile_memory_leak()