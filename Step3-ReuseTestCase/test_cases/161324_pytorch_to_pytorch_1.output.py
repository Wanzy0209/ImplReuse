import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_batch_isend_irecv_with_views(rank, world_size):
    """
    Test case to verify data consistency when using torch.distributed.isend
    (via batch_isend_irecv) with 2D tensor views.
    
    This test reproduces the scenario described in Issue 161324 where sending
    tensor views can lead to data inconsistencies.
    """
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    batch_size = 4
    total_columns = 10
    # Define offsets to create non-contiguous or specific slices
    # Slicing [:, 0:4] and [:, 6:10]
    split_offsets = [0, 4, 6, 10]
    tag1 = 0
    tag2 = 1

    if rank == 0:
        # Sender rank
        # Create a tensor with deterministic data to verify consistency
        local_tensor = torch.arange(batch_size * total_columns, dtype=torch.float32).view(batch_size, total_columns)
        
        # Create views (slices) of the original tensor
        dst_t1 = local_tensor[:, split_offsets[0]:split_offsets[1]]
        dst_t2 = local_tensor[:, split_offsets[2]:split_offsets[3]]
        
        process_group = dist.group.WORLD
        
        # Define P2POps using torch.distributed.isend
        send_op1 = dist.P2POp(dist.isend, dst_t1, 1, process_group, tag1)
        send_op2 = dist.P2POp(dist.isend, dst_t2, 1, process_group, tag2)
        
        # Batch send operations
        reqs = dist.batch_isend_irecv([send_op1, send_op2])
        
        # Wait for all operations to complete
        for req in reqs:
            req.wait()
            
        print(f"Rank {rank} sent tensors with shapes {dst_t1.shape} and {dst_t2.shape}")

    elif rank == 1:
        # Receiver rank
        local_tensor_dst = torch.zeros(batch_size, total_columns)
        
        tensor_col_offset1 = split_offsets[0]
        end_col_offset1 = split_offsets[1]
        tensor_col_offset2 = split_offsets[2]
        end_col_offset2 = split_offsets[3]
        
        # Create views (slices) of the destination tensor
        receiving_tensor_view1 = local_tensor_dst[:, tensor_col_offset1:end_col_offset1]
        receiving_tensor_view2 = local_tensor_dst[:, tensor_col_offset2:end_col_offset2]
        
        process_group = dist.group.WORLD
        
        # Define P2POps using torch.distributed.irecv
        receive_op1 = dist.P2POp(dist.irecv, receiving_tensor_view1, 0, process_group, tag1)
        receive_op2 = dist.P2POp(dist.irecv, receiving_tensor_view2, 0, process_group, tag2)
        
        # Batch receive operations
        reqs = dist.batch_isend_irecv([receive_op1, receive_op2])
        
        # Wait for all operations to complete
        for req in reqs:
            req.wait()

        # Verification: Reconstruct the expected data
        expected_tensor = torch.arange(batch_size * total_columns, dtype=torch.float32).view(batch_size, total_columns)
        expected_view1 = expected_tensor[:, split_offsets[0]:split_offsets[1]]
        expected_view2 = expected_tensor[:, split_offsets[2]:split_offsets[3]]

        # Assert that the received data matches the expected data
        # This assertion will fail if the data inconsistency bug (Issue 161324) is present.
        assert torch.equal(receiving_tensor_view1, expected_view1), "Data mismatch in receiving_tensor_view1"
        assert torch.equal(receiving_tensor_view2, expected_view2), "Data mismatch in receiving_tensor_view2"
        
        print(f"Rank {rank} received tensors successfully and verified data consistency.")

    dist.destroy_process_group()

def main():
    world_size = 2
    # Spawn processes to simulate multi-rank environment
    mp.spawn(test_batch_isend_irecv_with_views, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()