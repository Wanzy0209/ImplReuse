import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_send_object_list(rank, world_size):
    # Initialize distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use 'nccl' if CUDA is available to match the bug report's context, otherwise 'gloo'
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Recreate tensors from the bug report (GQA shapes: q has 32 heads, k/v have 8 heads)
    with torch.device(device):
        q = torch.randn([2, 32, 4096, 128], dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn([2, 8, 4096, 128], dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn([2, 8, 4096, 128], dtype=torch.bfloat16, requires_grad=True)

    # Adaptation: Compile the send_object_list function similar to how flex_attention was compiled
    # We use fullgraph=True and backend="inductor" to match the bug report's compilation settings
    compiled_send = torch.compile(dist.send_object_list, fullgraph=True, backend="inductor")

    if rank == 0:
        # Adapted call site: Send the list of tensors using the compiled function
        compiled_send([q, k, v], dst=1)
    else:
        # Receive the objects to verify the operation completes successfully
        recv_list = [None, None, None]
        dist.recv_object_list(recv_list, src=0)
        
        # Verify shapes to ensure data integrity
        assert recv_list[0].shape == q.shape, "Shape mismatch for q"
        assert recv_list[1].shape == k.shape, "Shape mismatch for k"
        assert recv_list[2].shape == v.shape, "Shape mismatch for v"

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Use spawn to run the distributed test
    mp.spawn(test_send_object_list, args=(world_size,), nprocs=world_size, join=True)