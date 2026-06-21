import torch
import sys

# Handle missing torch.export module (available in PyTorch 2.1+)
try:
    import torch.export
except ModuleNotFoundError:
    print("torch.export module not found. This test requires PyTorch >= 2.1. Skipping.")
    sys.exit(0)

class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor):
        out_0 = torch.zeros(2**20, device=inp.device, dtype=torch.float32)
        out_1 = torch.zeros(2**20, device=inp.device, dtype=torch.float32)
        ctx.save_for_backward(
            inp,
            out_0,
            out_1,
        )
        return out_0, out_1

    @staticmethod
    def backward(ctx, dA, dB):
        _ = ctx.saved_tensors  # this is necessary
        return None

class TestModule(torch.nn.Module):
    def forward(self, x):
        return MyOp.apply(x)[0]

if torch.cuda.is_available():
    dummy_input = torch.randn(2**20, device="cuda")
    model = TestModule().to("cuda")
    
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    
    # Adapted loop to test torch.export.export
    # Reduced iterations to 100 as export is heavier than checkpoint
    for i in range(100):
        # Replaced torch.utils.checkpoint.checkpoint with torch.export.export
        ep = torch.export.export(model, (dummy_input,))
        
        # Explicitly delete to allow garbage collection
        del ep
        
        if i % 10 == 0:
            print(i, torch.cuda.memory_allocated() / 1024**2, "MiB")
    
    end_mem = torch.cuda.memory_allocated()
    print(f"Final memory: {end_mem / 1024**2} MiB")
    print(f"Delta: {(end_mem - start_mem) / 1024**2} MiB")
    
    # Assertion to check for memory leaks
    # We expect some overhead, but not linear growth with iterations (e.g. > 100MB)
    assert (end_mem - start_mem) < 100 * 1024**2, "Memory leak detected in torch.export.export"
else:
    print("CUDA not available, skipping test.")