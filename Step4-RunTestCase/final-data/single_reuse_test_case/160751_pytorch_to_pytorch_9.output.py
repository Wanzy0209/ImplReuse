import torch
import torch.distributed as dist
import os

def setup():
    """Initialize the distributed process group."""
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        
        # The bug is specific to CUDA, so we prioritize NCCL backend
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            dist.init_process_group(backend='nccl', rank=0, world_size=1)
        else:
            # Fallback for environments without CUDA, though the bug context is CUDA
            dist.init_process_group(backend='gloo', rank=0, world_size=1)

def func():
    # Adapted call site using the similar API: torch.distributed.broadcast_object_list
    # We create a list of tensors to broadcast
    obj_list = [torch.tensor([1.0, -2.0], device="cuda")]
    
    # Call the similar API
    torch.distributed.broadcast_object_list(obj_list, src=0)
    
    # The synchronization call that was reported to be removed in aot_eager mode
    torch.cuda.synchronize()
    
    # This print statement serves as a control flow check
    print("Execution completed after broadcast and synchronize")

def test_fn():
    setup()
    
    # Check if torch._dynamo exists to avoid AttributeError in older PyTorch versions
    if hasattr(torch, '_dynamo'):
        torch._dynamo.reset()
    else:
        print("Skipping test: torch._dynamo is not available in this PyTorch version.")
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    
    # Compile with the specific backend mentioned in the bug report
    f_c = torch.compile(func, backend="aot_eager")
    
    try:
        f_c()
    except Exception as e:
        print(f"Test failed with exception: {e}")
    finally:
        # Cleanup distributed environment
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    test_fn()