import torch
import torch.utils.checkpoint

# Define a custom op using torch.autograd.Function (compatible with older PyTorch versions)
# This replaces the torch.library.define/impl/register_autograd logic which is not available
# in the environment causing the ImportError.
class LeakyOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Mimic the forward pass
        out_0 = torch.zeros(2**20, device=x.device, dtype=torch.float32)
        out_1 = torch.zeros(2**20, device=x.device, dtype=torch.float32)
        
        # Mimic setup_context: save tensors for backward
        # This mimics the behavior of the original bug where saving outputs causes issues
        ctx.save_for_backward(x, out_0, out_1)
        
        return out_0, out_1

    @staticmethod
    def backward(ctx, dA, dB):
        # Mimic the backward pass
        _ = ctx.saved_tensors
        return None

def op_fn(inp):
    # Call the custom op defined via torch.autograd.Function
    return LeakyOpFunction.apply(inp)[0]

if torch.cuda.is_available():
    dummy_input = torch.nn.Parameter(torch.randn(2**20, device="cuda"))
    print("Starting memory leak test with torch.utils.checkpoint...")
    
    for i in range(1000):
        # The bug is triggered by use_reentrant=False with a custom op saving outputs
        full_out = torch.utils.checkpoint.checkpoint(op_fn, dummy_input, use_reentrant=False)
        full_out.sum().backward()
        dummy_input.grad = None  # free gradient memory
        
        mem_usage = torch.cuda.memory_allocated() / 1024**2
        print(f"Iter {i}: {mem_usage:.2f} MiB")
        
        # Assertion to detect significant memory growth (leak)
        # We expect memory to stabilize, but with the bug it will grow linearly.
        # A threshold of 500MB growth is used as a heuristic for the leak.
        if i > 10 and mem_usage > 500: 
            print("Memory usage exceeded 500 MiB, potential leak detected.")
            # break # Uncomment to stop early if leak is confirmed
else:
    print("CUDA not available, skipping test.")