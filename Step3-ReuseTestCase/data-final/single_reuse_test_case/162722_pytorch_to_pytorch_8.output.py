import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import os

# Model definition from the original bug report
class CausalAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CausalAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Rank 0: Create model and data, then send
        model = CausalAttention(embed_size=256, heads=8)
        # Create dummy inputs
        values = keys = query = torch.randn(2, 10, 256)
        mask = torch.ones(2, 10, 10)
        
        # Run a forward pass to ensure buffers are initialized
        _ = model(values, keys, query, mask)
        
        # Prepare list of objects to send (Model state dict and inputs)
        # This tests sending complex picklable objects
        objects_to_send = [model.state_dict(), values, mask]
        
        print(f"Rank {rank}: Sending objects...")
        dist.send_object_list(objects_to_send, dst=1)
        
    elif rank == 1:
        # Rank 1: Receive objects and verify numerical consistency
        # The list must be pre-allocated with the correct size
        recv_buffer = [None] * 3 
        
        print(f"Rank {rank}: Receiving objects...")
        # Call the similar API: torch.distributed.recv_object_list
        dist.recv_object_list(recv_buffer, src=0)
        
        state_dict_recv, values_recv, mask_recv = recv_buffer
        
        # Reconstruct model from received state dict
        model_recv = CausalAttention(embed_size=256, heads=8)
        model_recv.load_state_dict(state_dict_recv)
        
        # Run inference with received data
        output = model_recv(values_recv, values_recv, values_recv, mask_recv)
        
        # Verify numerical consistency (check for NaNs/Infs which were the issue in the original bug)
        assert torch.isfinite(output).all(), "Numerical inconsistency detected: Output contains NaN or Inf"
        
        # Verify shape consistency
        assert output.shape == (2, 10, 256), f"Shape mismatch: expected (2, 10, 256), got {output.shape}"
        
        print(f"Rank {rank}: Test passed. Received objects are numerically consistent.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn 2 processes
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)