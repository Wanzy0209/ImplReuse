import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_batch_isend_irecv_views(rank, world_size):
    """
    Test case to verify data consistency when using batch_isend_irecv 
    with 2D tensor views. Uses torch.equal to check for data corruption.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    batch_size = 4
    total_columns = 10
    # Define offsets for two non-contiguous or distinct slices
    # Slice 1: columns 0 to 3
    # Slice 2: columns 5 to 8
    split_offsets = [0, 3, 5, 8]
    
    tag1 = 0
    tag2 = 1

    # Use a fixed seed for reproducibility across ranks
    torch.manual_seed(42)

    if rank == 0:
        # Sender rank
        local_tensor = torch.randn(batch_size, total_columns)
        
        # Create views to send
        dst_t1 = local_tensor[:, split_offsets[0]:split_offsets[1]]
        dst_t2 = local_tensor[:, split_offsets[2]:split_offsets[3]]
        
        process_group = dist.group.WORLD
        send_op1 = dist.P2POp(dist.isend, dst_t1, 1, process_group, tag1)
        send_op2 = dist.P2POp(dist.isend, dst_t2, 1, process_group, tag2)
        
        # Batch send operations
        reqs = dist.batch_isend_irecv([send_op1, send_op2])
        
        # Wait for completion
        for req in reqs:
            req.wait()
            
    elif rank == 1:
        # Receiver rank
        local_tensor_dst = torch.zeros(batch_size, total_columns)
        
        # Create views to receive into
        receiving_tensor_view1 = local_tensor_dst[:, split_offsets[0]:split_offsets[1]]
        receiving_tensor_view2 = local_tensor_dst[:, split_offsets[2]:split_offsets[3]]
        
        process_group = dist.group.WORLD
        receive_op1 = dist.P2POp(dist.irecv, receiving_tensor_view1, 0, process_group, tag1)
        receive_op2 = dist.P2POp(dist.irecv, receiving_tensor_view2, 0, process_group, tag2)
        
        # Batch receive operations
        reqs = dist.batch_isend_irecv([receive_op1, receive_op2])
        
        # Wait for completion
        for req in reqs:
            req.wait()

        # --- Verification using torch.equal ---
        
        # Regenerate the expected data on the receiver side
        torch.manual_seed(42)
        expected_tensor = torch.randn(batch_size, total_columns)
        expected_view1 = expected_tensor[:, split_offsets[0]:split_offsets[1]]
        expected_view2 = expected_tensor[:, split_offsets[2]:split_offsets[3]]

        # Check if the received views match the expected views
        # This assertion will fail if the data inconsistency bug exists
        assert torch.equal(receiving_tensor_view1, expected_view1), \
            f"Data mismatch in view 1. Expected:\n{expected_view1}\nGot:\n{receiving_tensor_view1}"
        
        assert torch.equal(receiving_tensor_view2, expected_view2), \
            f"Data mismatch in view 2. Expected:\n{expected_view2}\nGot:\n{receiving_tensor_view2}"

        print(f"Rank {rank}: Test passed. Data is consistent.")

    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(test_batch_isend_irecv_views, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()