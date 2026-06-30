import torch
import torch.utils.checkpoint
import unittest
import gc

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

def op_fn(inp):
    return MyOp.apply(inp)[0]

class TestCheckpointMemoryLeak(unittest.TestCase):
    def test_custom_autograd_checkpoint_memory_leak(self):
        # Leveraging the similar API pattern: checking backend availability
        # We reuse the pattern of torch.backends.nnpack.is_available() to ensure
        # the environment is ready for testing, even though the specific backend
        # differs. This reflects the code similarity in usage patterns.
        # Fix: Use hasattr to check if nnpack exists, as it was removed in newer PyTorch versions
        if hasattr(torch.backends, 'nnpack') and not torch.backends.nnpack.is_available():
            print("NNPACK backend not available, skipping backend-specific pattern check.")

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Run garbage collection before starting to ensure a clean state
        gc.collect()
        torch.cuda.empty_cache()

        dummy_input = torch.nn.Parameter(torch.randn(2**20, device="cuda"))
        
        # Track initial memory
        initial_mem = torch.cuda.memory_allocated()
        print(f"Initial Memory: {initial_mem / 1024**2:.2f} MiB")

        # Run the loop that triggers the leak in the bug report
        for i in range(100):
            full_out = torch.utils.checkpoint.checkpoint(op_fn, dummy_input, use_reentrant=False)
            full_out.sum().backward()
            dummy_input.grad = None  # free gradient memory
            
            if i % 10 == 0:
                current_mem = torch.cuda.memory_allocated()
                print(f"Iter {i}: {current_mem / 1024**2:.2f} MiB")

        final_mem = torch.cuda.memory_allocated()
        print(f"Final Memory: {final_mem / 1024**2:.2f} MiB")
        
        # In a fixed version, memory growth should be minimal (bounded by the loop size).
        # We assert that memory growth is less than 500MB as a sanity check for the fix.
        # The original bug would show unbounded growth (e.g., > 10GB).
        memory_growth = final_mem - initial_mem
        self.assertLess(memory_growth, 500 * 1024**2, 
                        f"Memory leak detected: Growth of {memory_growth/1024**2:.2f} MiB")

if __name__ == '__main__':
    unittest.main()