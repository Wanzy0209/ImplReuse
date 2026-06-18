import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run(rank, world_size):
    # Initialize distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create test tensors
    # Using CPU for distributed communication as MPS support is limited/complex
    W = torch.randn(12, 64, 768)
    
    # Create non-contiguous weight via permute (similar to rearrange)
    # Original: "h d m -> m (h d)"
    # Permute to (m, h, d) then reshape to (m, h*d)
    w_noncontig = W.permute(2, 0, 1).reshape(768, -1)
    w_contig = w_noncontig.contiguous()

    print(f"Rank {rank}: Weight contiguous: {w_contig.is_contiguous()}")
    print(f"Rank {rank}: Weight non-contiguous: {w_noncontig.is_contiguous()}")

    if rank == 0:
        # Send non-contiguous tensor
        req1 = dist.isend(w_noncontig, dst=1, tag=1)
        # Send contiguous tensor
        req2 = dist.isend(w_contig, dst=1, tag=2)
        req1.wait()
        req2.wait()
    elif rank == 1:
        # Receive tensors
        recv_noncontig = torch.empty_like(w_noncontig)
        recv_contig = torch.empty_like(w_contig)

        dist.recv(recv_noncontig, src=0, tag=1)
        dist.recv(recv_contig, src=0, tag=2)

        # Verify results
        # The bug in linear was that the result was wrong.
        # Here, we check if the received data matches the sent data.
        match_noncontig = torch.allclose(recv_noncontig, w_noncontig, atol=1e-5)
        match_contig = torch.allclose(recv_contig, w_contig, atol=1e-5)

        print(f"Rank 1: Contiguous match: {match_contig}")
        print(f"Rank 1: Non-contiguous match: {match_noncontig}")

        if not match_noncontig:
            print("BUG: isend failed to correctly transmit non-contiguous tensor data.")
            print(f"Max difference: {torch.abs(recv_noncontig - w_noncontig).max()}")
        else:
            print("SUCCESS: isend correctly handled non-contiguous tensor.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)