import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run_test(rank, world_size):
    """
    Worker function for distributed testing.
    Adapted to use bfloat16 tensors and configurations similar to the meta-llama bug report.
    """
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # Sender
        # Create a list of objects to send. 
        # Based on the bug report, we use bfloat16 tensors and a configuration dict.
        # Simulating logits or hidden states with shape (1, 1000) similar to input_ids in the bug.
        tensor_list = [torch.randn(1, 1000, dtype=torch.bfloat16) for _ in range(5)]
        
        # Simulating generation_config from the bug report
        config_dict = {
            "do_sample": False,
            "use_cache": True,
            "temperature": 0.0,
            "pad_token_id": 0
        }
        
        object_list = tensor_list + [config_dict]

        # Call the API under test
        dist.send_object_list(object_list, dst=1)
        print(f"Rank {rank}: Successfully sent object list containing bfloat16 tensors.")

    else:
        # Receiver
        # Prepare a list to receive objects. Size must match the sender.
        recv_list = [None] * 6 

        # Receive objects
        dist.recv_object_list(recv_list, src=0)

        # Verification
        assert len(recv_list) == 6, "Received list length mismatch"

        # Verify tensor properties (bfloat16 context from the bug)
        for i in range(5):
            assert isinstance(recv_list[i], torch.Tensor), f"Item {i} is not a Tensor"
            assert recv_list[i].dtype == torch.bfloat16, f"Item {i} is not bfloat16"
            assert recv_list[i].shape == (1, 1000), f"Item {i} shape mismatch"

        # Verify config properties
        assert isinstance(recv_list[5], dict), "Config item is not a dictionary"
        assert recv_list[5]["temperature"] == 0.0, "Config value mismatch"
        assert recv_list[5]["use_cache"] == True, "Config value mismatch"

        print(f"Rank {rank}: Successfully received and verified object list.")

    dist.destroy_process_group()

def test_distributed_send_object_list():
    """
    Test case for torch.distributed.send_object_list.
    Verifies that bfloat16 tensors and configuration objects are correctly sent and received.
    """
    world_size = 2
    # Use spawn to launch processes
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    test_distributed_send_object_list()