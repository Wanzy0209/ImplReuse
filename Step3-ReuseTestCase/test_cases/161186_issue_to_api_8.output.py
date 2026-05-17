import torch
import torch.utils.checkpoint
import unittest
import gc

class TestCheckpointMemoryLeak(unittest.TestCase):
    def test_custom_autograd_function_memory_leak(self):
        """
        Test for memory leak when a custom autograd Function is used with
        torch.utils.checkpoint.checkpoint (use_reentrant=False).
        
        The leak occurs because AC raises an exception to exit early, 
        bypassing the cleanup of saved tensors in the custom function.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        class MyOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp: torch.Tensor):
                # Create large tensors to make memory leak obvious
                out_0 = torch.zeros(2**20, device=inp.device, dtype=torch.float32)
                out_1 = torch.zeros(2**20, device=inp.device, dtype=torch.float32)
                # Saving outputs for backward is necessary to trigger the specific leak path
                ctx.save_for_backward(inp, out_0, out_1)
                return out_0, out_1

            @staticmethod
            def backward(ctx, dA, dB):
                _ = ctx.saved_tensors
                return None

        def op_fn(inp):
            return MyOp.apply(inp)[0]

        dummy_input = torch.nn.Parameter(torch.randn(2**20, device="cuda"))
        
        # Warm up
        full_out = torch.utils.checkpoint.checkpoint(op_fn, dummy_input, use_reentrant=False)
        full_out.sum().backward()
        dummy_input.grad = None
        torch.cuda.empty_cache()
        gc.collect()

        initial_mem = torch.cuda.memory_allocated()
        
        # Run multiple iterations to detect leak
        iterations = 10
        for i in range(iterations):
            full_out = torch.utils.checkpoint.checkpoint(op_fn, dummy_input, use_reentrant=False)
            full_out.sum().backward()
            dummy_input.grad = None
            
            # Force cleanup to ensure we are measuring retained references, not just pending GC
            del full_out
            gc.collect()
        
        final_mem = torch.cuda.memory_allocated()
        mem_diff = final_mem - initial_mem
        
        # The leak is significant (megabytes per iteration). 
        # We allow a small buffer for noise, but expect the diff to be small if fixed.
        # If the bug exists, mem_diff will be roughly iterations * size_of_tensors.
        # 2 tensors of 2**20 floats ~ 8MB. 10 iterations -> ~80MB leak.
        self.assertLess(mem_diff, 10 * 1024 * 1024, 
                        f"Memory leak detected: {mem_diff / 1024**2:.2f} MiB increased over {iterations} iterations")

if __name__ == '__main__':
    unittest.main()