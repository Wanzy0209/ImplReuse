import torch
import torch.distributed as dist
import os
import multiprocessing as mp

def run_worker(local_rank):
    """
    Worker function to test the all_to_all operation.
    Returns True if successful, False if the specific 'not supported' error occurs.
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = '2'
    
    try:
        # Initialize process group
        dist.init_process_group(backend='gloo', rank=local_rank, world_size=2)
        
        # Prepare tensors for all_to_all
        world_size = 2
        output = list(torch.empty([world_size], dtype=torch.int64).chunk(world_size))
        input_tensor = list((torch.arange(world_size) + local_rank * world_size).chunk(world_size))
        
        # Attempt the operation
        dist.all_to_all(output, input_tensor)
        
        # Cleanup
        dist.destroy_process_group()
        return True
    except RuntimeError as e:
        if "does not support alltoall" in str(e):
            return False
        # Re-raise if it's a different error
        raise

def is_gloo_all_to_all_available():
    """
    Checks if the GLOO backend supports all_to_all.
    This function mimics the pattern of torch.backends.mkldnn.is_available
    by determining the capability of the backend through execution.
    """
    world_size = 2
    # Use 'spawn' context to ensure compatibility across different OSs
    ctx = mp.get_context('spawn')
    
    with ctx.Pool(world_size) as pool:
        results = pool.map(run_worker, range(world_size))
    
    # The feature is available only if all workers succeed
    return all(results)

if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    
    # Check availability using the pattern similar to torch.backends.mkldnn.is_available
    supported = is_gloo_all_to_all_available()
    print(f"GLOO all_to_all supported: {supported}")
    
    # Assertion based on the bug report #162248
    # The bug report states that GLOO does not support all_to_all in v2.8.0
    assert not supported, "Expected GLOO to not support all_to_all based on bug report #162248"