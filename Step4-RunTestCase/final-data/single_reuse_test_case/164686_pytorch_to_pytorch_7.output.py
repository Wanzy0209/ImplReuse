import torch
import torch.distributed as dist

# Setup to match the original bug report context
torch.manual_seed(13653)

def generate_test_objects(arg_0, arg_1):
    # Replicating the logic from the fuzzed_program
    # Note: The bug report comments indicate specific dtypes (e.g., int32, int64),
    # but the Python code uses literals. We follow the Python code structure.
    var_node_3 = torch.full((), 1.0, dtype=torch.float32)
    var_node_2 = var_node_3.item()
    
    var_node_5 = -3
    var_node_6 = arg_0
    var_node_4 = var_node_5 + var_node_6
    
    var_node_1 = var_node_2 + var_node_4
    
    var_node_9 = 1
    var_node_10 = -10
    var_node_8 = var_node_9 / var_node_10
    
    var_node_12 = arg_1
    var_node_13 = -5
    var_node_11 = var_node_12 / var_node_13
    
    var_node_7 = var_node_8 + var_node_11
    var_node_0 = var_node_1 * var_node_7
    
    # We return the intermediate values to test serialization of various types
    return [var_node_0, var_node_1, var_node_4, var_node_7, var_node_8, var_node_11]

# Generate inputs
arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
arg_1 = torch.tensor(torch.randn(()), dtype=torch.int64).item()

# Generate the objects to be sent
object_list = generate_test_objects(arg_0, arg_1)

# Adaptation: Replace torch.compile call with torch.distributed.send_object_list
# We check if distributed is initialized to ensure the test is runnable in non-distributed environments
if dist.is_available() and dist.is_initialized():
    try:
        # Determine a valid destination rank (e.g., next rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dst = (rank + 1) % world_size
        
        # Call the similar API
        dist.send_object_list(object_list, dst=dst)
        print(' send_object_list success')
    except Exception as e:
        print(f' send_object_list failed: {e}')
else:
    # Fallback for environments without distributed setup:
    # Verify that the objects are picklable, which is a requirement for send_object_list.
    # This ensures the test logic is validated even without a process group.
    import pickle
    try:
        pickle.dumps(object_list)
        print(' Objects are picklable (send_object_list prerequisite met)')
    except Exception as e:
        print(f' Objects are not picklable: {e}')