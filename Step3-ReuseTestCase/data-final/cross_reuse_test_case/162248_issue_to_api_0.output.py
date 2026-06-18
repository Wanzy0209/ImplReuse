import torch
import torch.distributed as dist
import os
import multiprocessing as mp

def run(local_rank):
    # Initialize the process group with GLOO backend
    dist.init_process_group(backend='gloo', rank=local_rank)
    
    # Leverage the similar API: torch.distributed.get_world_size
    # This replaces the hardcoded 'world_size' variable used in the original reproducer.
    # It allows the tensor dimensions to adapt to the actual distributed environment size.
    ws = dist.get_world_size()
    
    # Prepare input and output tensors based on the dynamic world size
    input_list = list((torch.arange(ws, dtype=torch.int64) + local_rank * ws).chunk(ws))
    output_list = list(torch.empty([ws], dtype=torch.int64).chunk(ws))
    
    # Test the API under test: torch.distributed.all_to_all
    # Based on the bug report, this is expected to raise RuntimeError for GLOO
    try:
        dist.all_to_all(output_list, input_list)
        print(f"Rank {local_rank}: all_to_all succeeded (unexpected if bug exists)")
    except RuntimeError as e:
        # Verify the specific error message related to the bug
        assert "does not support alltoall" in str(e), f"Unexpected error: {e}"
        print(f"Rank {local_rank}: Caught expected error - {e}")

if __name__ == "__main__":
    print(f"PyTorch Version: {torch.__version__}")
    
    world_size = 2
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Ensure spawn context is used for compatibility
    mp.set_start_method('spawn', force=True)
    
    with mp.Pool(world_size) as pool:
        pool.map(run, range(world_size))