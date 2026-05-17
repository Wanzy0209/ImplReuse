import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Helper function inspired by the pattern of tf.executing_eagerly
# to check the state of the tensor (view vs contiguous).
def is_tensor_view(tensor):
    """
    Checks whether the tensor is a non-contiguous view.
    This mimics the state-checking pattern of tf.executing_eagerly.
    """
    return not tensor.is_contiguous()

def run_test(rank, world_size):
    # Initialize process group
    dist.init_process_group(
        backend='gloo', # Using gloo for CPU compatibility
        init_method=f'tcp://127.0.0.1:12345',
        rank=rank,
        world_size=world_size
    )

    # Set seed for reproducibility
    torch.manual_seed(42)
    
    batch_size = 4
    total_columns = 10
    # Define split offsets: [0, 3, 5, 10] -> slices 0:3 and 5:10
    split_offsets = [0, 3, 5, 10]

    if rank == 0:
        # Sender rank
        local_tensor = torch.randn(batch_size, total_columns)
        
        # Create views (slices)
        dst_t1 = local_tensor[:, split_offsets[0]:split_offsets[1]]
        dst_t2 = local_tensor[:, split_offsets[2]:split_offsets[3]]

        # Verify we are using views (mimicking the state check pattern)
        assert is_tensor_view(dst_t1), "dst_t1 should be a view"
        assert is_tensor_view(dst_t2), "dst_t2 should be a view"

        process_group = dist.group.WORLD
        tag1, tag2 = 0, 1
        
        send_op1 = dist.P2POp(
            dist.isend, dst_t1, 1, process_group, tag1
        )
        send_op2 = dist.P2POp(
            dist.isend, dst_t2, 1, process_group, tag2
        )
        
        # Batch send operations
        reqs = dist.batch_isend_irecv([send_op1, send_op2])
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
        
        # Create views (slices) into the destination tensor
        receiving_tensor_view1 = local_tensor_dst[:, tensor_col_offset1:end_col_offset1]
        receiving_tensor_view2 = local_tensor_dst[:, tensor_col_offset2:end_col_offset2]

        # Verify we are using views
        assert is_tensor_view(receiving_tensor_view1), "receiving_tensor_view1 should be a view"
        assert is_tensor_view(receiving_tensor_view2), "receiving_tensor_view2 should be a view"

        process_group = dist.group.WORLD
        tag1, tag2 = 0, 1
        
        receive_op1 = dist.P2POp(
            dist.irecv, receiving_tensor_view1, 0, process_group, tag1
        )
        receive_op2 = dist.P2POp(
            dist.irecv, receiving_tensor_view2, 0, process_group, tag2
        )
        
        # Batch receive operations
        reqs = dist.batch_isend_irecv([receive_op1, receive_op2])
        for req in reqs:
            req.wait()

        # Generate expected tensor using the same seed
        torch.manual_seed(42)
        expected_tensor = torch.randn(batch_size, total_columns)
        
        # Check for data consistency
        # The bug report indicates data inconsistencies when using views.
        # We check if the received data matches the expected data.
        
        # Extract the specific slices from expected to compare directly
        expected_view1 = expected_tensor[:, tensor_col_offset1:end_col_offset1]
        expected_view2 = expected_tensor[:, tensor_col_offset2:end_col_offset2]

        match1 = torch.allclose(receiving_tensor_view1, expected_view1)
        match2 = torch.allclose(receiving_tensor_view2, expected_view2)

        if not (match1 and match2):
            print(f"Rank {rank} DATA INCONSISTENCY DETECTED!")
            print(f"View 1 Match: {match1}, View 2 Match: {match2}")
            print(f"Received View 1:\n{receiving_tensor_view1}")
            print(f"Expected View 1:\n{expected_view1}")
            print(f"Received View 2:\n{receiving_tensor_view2}")
            print(f"Expected View 2:\n{expected_view2}")
            # In a real test framework, this would be assert False
        else:
            print(f"Rank {rank} received data correctly.")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)