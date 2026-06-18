import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use 'gloo' backend for CPU compatibility to match the original bug's device context
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)

    # Replicate the tensor setup from the original bug report
    # dividend contains INT64_MIN
    dividend = torch.full((2, 3), torch.iinfo(torch.int64).min, dtype=torch.int64, device='cpu')
    # divisor contains -1
    divisor = torch.full((3,), -1, dtype=torch.int64, device='cpu')

    print(f"Rank {rank} - Dividend tensor:", dividend)
    print(f"Rank {rank} - Divisor tensor:", divisor)

    # Adaptation for torch.distributed.reduce_scatter
    # reduce_scatter takes a list of output tensors and a list of input tensors.
    # We adapt the 'dividend' tensor to be the input list.
    # The operation reduces (sums by default) the inputs from all ranks and scatters the result.
    
    # Input list: containing the problematic INT64_MIN tensor
    input_list = [dividend]
    
    # Output list: must be pre-allocated. 
    # Since input is (2, 3) and world_size is 2, output on each rank will be (1, 3).
    output_list = [torch.empty((1, 3), dtype=torch.int64, device='cpu')]

    try:
        # Call the similar API
        dist.reduce_scatter(output_list, input_list)
        print(f"Rank {rank} - Result:", output_list[0])
    except Exception as e:
        print(f"Rank {rank} - Exception occurred: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn processes to simulate a distributed environment
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)