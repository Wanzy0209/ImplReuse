import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_send_recv_views(rank, world_size):
    """
    Test case to verify data consistency when using torch.distributed.send/recv
    with 2D tensor views, adapted from the batch_isend_irecv bug report.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    batch_size = 4
    total_columns = 10
    # Define slices: columns [0:3] and [7:10]
    split_offsets = [0, 3, 7, 10]

    if rank == 0:
        # Sender rank
        local_tensor = torch.randn(batch_size, total_columns)
        
        # Create 2D tensor views (slices)
        send_view1 = local_tensor[:, split_offsets[0]:split_offsets[1]]
        send_view2 = local_tensor[:, split_offsets[2]:split_offsets[3]]

        # Adaptation: Use torch.distributed.send (Synchronous) instead of batch_isend_irecv/isend
        # We send the views directly
        dist.send(send_view1, dst=1)
        dist.send(send_view2, dst=1)
        
        # Send the full tensor separately to verify correctness on the receiver side
        dist.send(local_tensor, dst=1, tag=999)

    elif rank == 1:
        # Receiver rank
        local_tensor_dst = torch.zeros(batch_size, total_columns)
        
        # Create 2D tensor views (slices) matching the sender's layout
        recv_view1 = local_tensor_dst[:, split_offsets[0]:split_offsets[1]]
        recv_view2 = local_tensor_dst[:, split_offsets[2]:split_offsets[3]]

        # Adaptation: Use torch.distributed.recv (Synchronous) instead of batch_isend_irecv/irecv
        # We receive directly into the views
        dist.recv(recv_view1, src=0)
        dist.recv(recv_view2, src=0)
        
        # Receive the expected full tensor for comparison
        expected_tensor = torch.zeros(batch_size, total_columns)
        dist.recv(expected_tensor, src=0, tag=999)

        # Assertion: Check if the data received into the views matches the source data
        # This detects the "Data inconsistencies" described in the bug report
        assert torch.equal(local_tensor_dst, expected_tensor), \
            f"Data mismatch! Rank 1 received inconsistent data in tensor views.\nExpected:\n{expected_tensor}\nGot:\n{local_tensor_dst}"

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_send_recv_views, args=(world_size,), nprocs=world_size, join=True)