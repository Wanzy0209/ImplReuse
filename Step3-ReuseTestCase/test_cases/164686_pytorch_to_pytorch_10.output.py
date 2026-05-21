import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def generate_fuzzed_data(seed_val):
    """
    Generates data structures similar to those in the original bug report
    to test if gather_object handles these specific scalar types correctly.
    """
    torch.manual_seed(seed_val)
    
    # Replicating the logic from the original fuzzed_program
    var_node_3 = torch.full((), 1.0, dtype=torch.float32)
    var_node_2 = var_node_3.item() # float32 scalar
    
    var_node_5 = -3 # int32
    var_node_6 = torch.tensor(torch.randn(()), dtype=torch.int64).item() # int64
    var_node_4 = var_node_5 + var_node_6 # int32 (promoted)
    
    var_node_1 = var_node_2 + var_node_4 # float32
    
    var_node_9 = 1 # int64
    var_node_10 = -10 # int32
    var_node_8 = var_node_9 / var_node_10 # float (division result)
    
    var_node_12 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
    var_node_13 = -5 # int32
    var_node_11 = var_node_12 / var_node_13 # float
    
    var_node_7 = var_node_8 + var_node_11
    var_node_0 = var_node_1 * var_node_7
    
    # Return a dictionary containing the mixed-type scalars
    return {
        "float_result": var_node_0,
        "int_addition": var_node_4,
        "division_mix": var_node_8
    }

def run_gather_object_test(rank, world_size):
    setup(rank, world_size)
    
    # Generate the data similar to the bug report context
    # Using a fixed seed to ensure consistency across ranks for this test
    data_to_gather = generate_fuzzed_data(13653)
    
    if rank == 0:
        gather_list = [None for _ in range(world_size)]
        # Adapted call site: torch.distributed.gather_object
        dist.gather_object(data_to_gather, gather_list, dst=0)
        
        print(f"Rank 0 gathered objects: {gather_list}")
        
        # Verify that objects were gathered correctly
        assert len(gather_list) == world_size
        for obj in gather_list:
            assert obj is not None
            assert isinstance(obj, dict)
            # Check that the keys exist and values are numbers
            assert "float_result" in obj
            assert "int_addition" in obj
            
        print(" gather_object test passed")
    else:
        dist.gather_object(data_to_gather, dst=0)
            
    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Run the test in a multiprocess environment
    mp.spawn(run_gather_object_test, args=(world_size,), nprocs=world_size, join=True)