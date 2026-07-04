import torch
import torch.distributed as dist
import os

def setup():
    # Minimal setup for single-process distributed testing
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def func(obj, dst=0):
    # Adapted function using torch.distributed.gather_object
    # Replaces the original jacfunc logic
    output = [None] * dist.get_world_size() if dist.get_rank() == dst else None
    dist.gather_object(obj, output, dst=dst)
    return output

if __name__ == "__main__":
    setup()

    # Prepare inputs similar to the original bug report (tensors)
    # gather_object accepts picklable objects, so we wrap the tensor
    input_obj = {"tensor": torch.rand((3, 3), dtype=torch.float64)}

    # 1. Test eager execution (works)
    print("Testing eager execution...")
    out_eager = func(input_obj)
    print(f"Eager result: {out_eager}")

    # 2. Test compiled execution (fails/works check)
    # Using dynamic=True as in the original bug report
    print("\nTesting compiled execution with dynamic=True...")
    
    # Fix: Check if torch.compile is available (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("torch.compile is not available in this PyTorch version. Skipping compiled test.")
    else:
        compiled_func = torch.compile(func, dynamic=True)
        
        try:
            out_compiled = compiled_func(input_obj)
            print(f"Compiled result: {out_compiled}")

            # Verify results
            if dist.get_rank() == 0:
                assert out_eager == out_compiled, "Outputs differ between eager and compiled"
                print("Test passed: Eager and Compiled outputs match.")
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()

    cleanup()