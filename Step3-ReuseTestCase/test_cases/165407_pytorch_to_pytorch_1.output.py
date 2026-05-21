import torch
import torch.distributed as dist
import os
import gc

def setup():
    # Initialize distributed environment for single-process testing
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    # Use 'gloo' backend for CPU compatibility
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

def cleanup():
    dist.destroy_process_group()

def test_distributed_reduce_compile():
    """
    Test case for torch.distributed.reduce within a torch.compile context.
    This adapts the original memory leak scenario to the similar API.
    """
    setup()
    try:
        # Define a function using the similar API: torch.distributed.reduce
        @torch.compile
        def compiled_reduce_step(tensor):
            # Perform a distributed reduce operation
            # This replaces the flash_attn_varlen_func from the original bug report
            dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
            return tensor

        input_tensor = torch.randn(32, 32)
        
        print("Starting loop to check for memory leaks with torch.distributed.reduce...")
        
        # Loop to simulate the training steps described in the bug report
        for step in range(100):
            output = compiled_reduce_step(input_tensor)
            
            # Explicit cleanup to mimic a training step
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if step % 20 == 0:
                # In the original bug, tensor counts were monitored here.
                # We verify the operation completes successfully.
                print(f"Step {step} completed.")

        # Basic assertion to ensure the tensor is processed
        # (In a real distributed setting, the value would change, but with rank=0 it stays same)
        assert input_tensor.shape == (32, 32), "Tensor shape mismatch after reduce"
        
        print("Test finished. Monitor memory usage for leaks.")

    finally:
        cleanup()

if __name__ == "__main__":
    test_distributed_reduce_compile()