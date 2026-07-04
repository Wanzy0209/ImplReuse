import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import os
import pickle

# Helper functions to replace missing recv_object_list/send_object_list
# These are necessary for PyTorch versions < 1.8 where these APIs do not exist.
def send_object_via_tensor(obj, dst):
    """Send an arbitrary picklable object to a destination rank."""
    buffer = pickle.dumps(obj)
    # Create a byte tensor from the buffer
    # Using frombuffer is efficient and available in older PyTorch versions
    byte_arr = bytearray(buffer)
    tensor = torch.frombuffer(byte_arr, dtype=torch.uint8)
    
    # Send the size of the tensor first so the receiver knows how much to allocate
    size_tensor = torch.tensor([tensor.numel()], dtype=torch.long)
    dist.send(size_tensor, dst=dst)
    
    # Send the actual tensor data
    dist.send(tensor, dst=dst)

def recv_object_via_tensor(src):
    """Receive an arbitrary picklable object from a source rank."""
    # Receive the size of the incoming tensor
    size_tensor = torch.tensor([0], dtype=torch.long)
    dist.recv(size_tensor, src=src)
    size = size_tensor.item()
    
    # Allocate buffer and receive tensor
    tensor = torch.empty(size, dtype=torch.uint8)
    dist.recv(tensor, src=src)
    
    # Convert back to bytes and deserialize
    buffer = bytes(tensor.numpy().tobytes())
    return pickle.loads(buffer)

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
        # Use the custom helper function instead of dist.send_object_list
        send_object_via_tensor(objects_to_send, dst=1)
        
    elif rank == 1:
        # Rank 1: Receive objects and verify numerical consistency
        print(f"Rank {rank}: Receiving objects...")
        # Use the custom helper function instead of dist.recv_object_list
        recv_buffer = recv_object_via_tensor(src=0)
        
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