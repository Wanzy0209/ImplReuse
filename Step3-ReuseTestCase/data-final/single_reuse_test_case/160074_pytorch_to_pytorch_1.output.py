import torch
import torch.distributed as dist
import os

def setup():
    # Initialize environment for single-process distributed execution
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    # Use 'gloo' backend for compatibility in testing environments
    dist.init_process_group(backend="gloo", world_size=1, rank=0)

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    setup()

    # Recreate tensors from the original bug report
    # Using CPU to ensure the test runs in any environment, 
    # though the original bug was on CUDA (B200).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with torch.device(device):
        q = torch.randn([2, 32, 4096, 128], dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn([2, 8, 4096, 128], dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn([2, 8, 4096, 128], dtype=torch.bfloat16, requires_grad=True)

    # Adapt the test to use torch.distributed.reduce
    # We wrap it in torch.compile to verify compilation stability, 
    # mirroring the context of the original bug.
    compiled_reduce = torch.compile(dist.reduce, fullgraph=True)

    try:
        # Perform reduce operation on the tensors
        # Since world_size=1, reducing to rank 0 is effectively a no-op 
        # but exercises the compilation and execution path.
        compiled_reduce(q, dst=0)
        compiled_reduce(k, dst=0)
        compiled_reduce(v, dst=0)
        
        # Assertions to verify tensors are still valid after operation
        assert q.shape == (2, 32, 4096, 128)
        assert k.shape == (2, 8, 4096, 128)
        assert v.shape == (2, 8, 4096, 128)
        
        print("Test passed: torch.distributed.reduce compiled and executed successfully.")
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        cleanup()