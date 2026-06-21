import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run(rank, world_size):
    """
    Worker function to be executed in each process.
    Defined at module level to be picklable for multiprocessing.
    """
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Set seed for reproducibility to allow value comparison
    torch.manual_seed(42)
    
    if rank == 0:
        # Sender process
        # Create a bfloat16 tensor, matching the dtype in the bug report
        tensor_to_send = torch.randn(10, 10, dtype=torch.bfloat16)
        dist.send_object_list([tensor_to_send], dst=1)
    else:
        # Receiver process
        # Generate the expected tensor locally for comparison
        expected_tensor = torch.randn(10, 10, dtype=torch.bfloat16)
        
        received_list = [None]
        
        # Define the function containing the API under test
        def recv_fn(obj_list):
            dist.recv_object_list(obj_list, src=0)
        
        # Adaptation: Wrap the API call in torch.compile, similar to the original test case
        compiled_recv = torch.compile(recv_fn)
        
        # Execute the compiled function
        compiled_recv(received_list)
        
        # Verify the received object
        # Replaced self.assert... with standard assert for multiprocessing compatibility
        assert received_list[0] is not None
        assert isinstance(received_list[0], torch.Tensor)
        
        # Verify correctness (values and dtype)
        # Note: bfloat16 has lower precision, so we use a tolerance similar to the original test
        assert torch.allclose(received_list[0], expected_tensor, atol=1e-5)
        assert received_list[0].dtype == torch.bfloat16

    dist.destroy_process_group()

class TestDistributedRecvObjectList(unittest.TestCase):
    def test_recv_object_list_compile_bfloat16(self):
        """
        Test case for torch.distributed.recv_object_list, adapted from the 
        meta-llama correctness issue (Issue 161457).
        
        The original issue highlights a correctness problem with torch.compile 
        and bfloat16. This test verifies that torch.distributed.recv_object_list
        works correctly when receiving bfloat16 tensors, even when the receiving
        logic is wrapped in torch.compile.
        """
        # Spawn 2 processes for the test
        mp.spawn(run, args=(2,), nprocs=2)

if __name__ == '__main__':
    unittest.main()