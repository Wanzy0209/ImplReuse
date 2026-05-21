import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use 'gloo' backend for CPU-based object broadcasting to ensure 
    # the test runs without requiring specific GPU hardware (like B200).
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()

def test_compiled_broadcast(rank, world_size):
    """
    Test case for torch.distributed.broadcast_object_list.
    Adapted to verify the API under the context of torch.compile regression.
    """
    setup(rank, world_size)

    # Define the function containing the similar API call.
    # We wrap this to test if torch.compile handles it correctly 
    # (checking for compilation regressions similar to the original issue).
    def broadcast_fn(object_list):
        dist.broadcast_object_list(object_list, src=0)
        return object_list

    # Apply torch.compile to the function
    compiled_broadcast_fn = torch.compile(broadcast_fn)

    # Prepare data: Rank 0 has the data, others have None
    if rank == 0:
        input_list = [1, "test_string", {"key": torch.tensor([1, 2, 3])}]
    else:
        input_list = [None, None, None]

    # Execute the compiled function
    result_list = compiled_broadcast_fn(input_list)

    # Verify correctness on all ranks
    expected_list = [1, "test_string", {"key": torch.tensor([1, 2, 3])}]
    
    for i in range(len(expected_list)):
        expected = expected_list[i]
        result = result_list[i]
        
        if isinstance(expected, torch.Tensor):
            assert torch.equal(result, expected), f"Rank {rank}: Tensor mismatch at index {i}"
        elif isinstance(expected, dict):
            # Handle dictionary comparison (specifically for tensors inside)
            assert result.keys() == expected.keys(), f"Rank {rank}: Dict keys mismatch"
            for k in expected:
                if isinstance(expected[k], torch.Tensor):
                    assert torch.equal(result[k], expected[k]), f"Rank {rank}: Dict tensor value mismatch for key {k}"
                else:
                    assert result[k] == expected[k], f"Rank {rank}: Dict value mismatch for key {k}"
        else:
            assert result == expected, f"Rank {rank}: Value mismatch at index {i}"

    print(f"Rank {rank}: Test passed successfully.")
    cleanup()

def main():
    world_size = 2
    # Spawn processes to simulate a distributed environment
    mp.spawn(test_compiled_broadcast, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()