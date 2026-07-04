import torch
import torch.distributed as dist
import os

def func():
    # Adapted to use torch.distributed.send_object_list
    # Note: This requires a distributed environment (e.g., torchrun) to execute the send.
    if dist.is_initialized():
        a = torch.tensor([1.0, -2.0], device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Check condition similar to original test case
        result = torch.all(a > 0)
        # assert result, "should throw"
        
        # Call the similar API
        dist.send_object_list([a], dst=0)
        
        # The critical part: synchronize call that might be removed in aot_eager
        torch.cuda.synchronize()
        
        print("should not run if error occurred")
    else:
        print("Distributed not initialized, skipping send.")

def test_fn():
    # Fix: Check if _dynamo attribute exists before calling reset
    if hasattr(torch, '_dynamo'):
        torch._dynamo.reset()
    
    # Compile with aot_eager to verify if synchronize is preserved
    f_c = torch.compile(func, backend="aot_eager")
    f_c()

if __name__ == "__main__":
    # Minimal initialization for demonstration
    if 'RANK' in os.environ:
        dist.init_process_group(backend='gloo')
    test_fn()