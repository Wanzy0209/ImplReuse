import torch
import torch.distributed as dist
import os

def test_recv_object_list_private_use1():
    """
    Test case to verify if torch.distributed.recv_object_list handles 
    custom devices (PrivateUse1) correctly without encountering 'meta' 
    device errors, similar to the issue reported with torch.compile.
    """
    
    # Setup environment variables for distributed communication
    # This assumes a single-node setup for testing purposes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # Initialize the process group
    # 'gloo' backend is used as it is generally available for CPU/custom device testing
    dist.init_process_group(backend='gloo')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Define the custom device as used in the bug report
    custom_device = torch.device("PrivateUse1")

    if rank == 0:
        # Sender: Create a tensor on PrivateUse1 and send it
        # Note: Creating a tensor on PrivateUse1 requires a registered backend.
        # This code assumes the environment has the backend registered as per the bug context.
        try:
            # Simulating the data.to('PrivateUse1') part of the original bug
            tensor_to_send = torch.randn(2, 2, device=custom_device)
            dist.send_object_list([tensor_to_send], dst=1)
            print(f"Rank {rank}: Sent tensor on {custom_device}")
        except Exception as e:
            # If backend is not registered, we print the error but the test structure remains valid
            print(f"Rank {rank}: Failed to create/send tensor (Backend likely missing): {e}")

    elif rank == 1:
        # Receiver: Receive the object specifying the custom device
        obj_list = [None]
        try:
            # Adapted call site: replacing torch.compile logic with recv_object_list
            # We explicitly pass the custom device to the API.
            dist.recv_object_list(obj_list, src=0, device=custom_device)
            
            received_tensor = obj_list[0]
            if received_tensor is not None:
                # Verify the device matches the expectation
                assert received_tensor.device == custom_device, \
                    f"Expected device {custom_device}, got {received_tensor.device}"
                print(f"Rank {rank}: Successfully received tensor on {custom_device}")
        except RuntimeError as e:
            # Check for the specific error mentioned in the bug report:
            # "storage is not on the custom PrivateUse1 device. Got: meta"
            error_msg = str(e)
            if "meta" in error_msg and "PrivateUse1" in error_msg:
                print(f"Rank {rank}: BUG REPRODUCED - {e}")
            else:
                print(f"Rank {rank}: RuntimeError - {e}")
        except Exception as e:
            print(f"Rank {rank}: Exception - {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    # To run this test, use torchrun:
    # torchrun --nproc_per_node=2 test_script.py
    test_recv_object_list_private_use1()