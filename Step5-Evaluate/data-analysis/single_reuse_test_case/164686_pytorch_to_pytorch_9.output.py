import torch
import torch.distributed as dist
import tempfile
import os

def test_broadcast_object_list():
    """
    Test case for torch.distributed.broadcast_object_list using data generation
    logic from the original bug report (Issue 164686).
    """
    # Setup a single-process distributed environment for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        store = dist.FileStore(os.path.join(tmpdir, "store"), 1)
        dist.init_process_group(
            backend="gloo",
            rank=0,
            world_size=1,
            store=store
        )

        torch.manual_seed(13653)

        # Logic adapted from the fuzzed_program in the bug report
        # We remove the gradient/sentinel logic as it is not applicable to broadcast_object_list
        def generate_data(arg_0, arg_1):
            var_node_3 = torch.full((), 1.0, dtype=torch.float32)
            var_node_2 = var_node_3.item()
            var_node_5 = -3
            var_node_6 = arg_0
            var_node_4 = var_node_5 + var_node_6
            var_node_1 = var_node_2 + var_node_4
            var_node_9 = 1
            var_node_10 = -10
            # Note: In Python 3, division of ints returns float. 
            # The bug report comments suggest specific dtypes (int64/int32), 
            # but we follow the executable Python code provided.
            var_node_8 = var_node_9 / var_node_10 
            var_node_12 = arg_1
            var_node_13 = -5
            var_node_11 = var_node_12 / var_node_13
            var_node_7 = var_node_8 + var_node_11
            var_node_0 = var_node_1 * var_node_7
            return var_node_0

        # Generate inputs
        arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
        arg_1 = torch.tensor(torch.randn(()), dtype=torch.int64).item()

        # Generate the object to broadcast
        result = generate_data(arg_0, arg_1)
        
        # Prepare object list for broadcast_object_list
        object_list = [result]
        
        # Call the similar API: torch.distributed.broadcast_object_list
        # This replaces the torch.compile call site from the original bug
        try:
            dist.broadcast_object_list(object_list, src=0)
            print(f" broadcast_object_list success. Received: {object_list[0]}")
            
            # Verify the data integrity
            assert object_list[0] == result, "Broadcasted object does not match original"
        except Exception as e:
            print(f" broadcast_object_list failed: {e}")
            raise
        finally:
            dist.destroy_process_group()

if __name__ == "__main__":
    test_broadcast_object_list()