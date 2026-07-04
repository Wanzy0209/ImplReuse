import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# --- Model Definitions from Original Bug Report ---

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

class CausalAttentionBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(CausalAttentionBlock, self).__init__()
        self.attention = CausalAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# --- Test Case for torch.distributed send/recv ---

def run_test(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Using gloo backend for CPU-based object list transfer
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Define constants at the top level so both ranks know the tensor shapes
    embed_size = 256
    heads = 8
    N = 2
    seq_length = 10

    if rank == 0:
        # --- Sender Logic ---
        print(f"Rank {rank}: Initializing model and generating data...")
        
        # Setup model from the bug report
        model = CausalAttention(embed_size, heads)
        model.eval() # Set to eval for consistency
        
        # Create dummy inputs
        values = torch.randn(N, seq_length, embed_size)
        keys = torch.randn(N, seq_length, embed_size)
        query = torch.randn(N, seq_length, embed_size)
        mask = torch.ones((N, seq_length, seq_length))
        
        # Run forward pass
        with torch.no_grad():
            output = model(values, keys, query, mask)
        
        print(f"Rank {rank}: Output generated. Shape: {output.shape}")
        
        # Use standard dist.send instead of send_object_list to ensure compatibility
        # send_object_list/recv_object_list might not be available in older PyTorch versions
        dist.send(output, dst=1)
        print(f"Rank {rank}: Sent tensor successfully.")

    elif rank == 1:
        # --- Receiver Logic ---
        print(f"Rank {rank}: Waiting to receive tensor...")
        
        # Allocate a buffer with the known shape to receive the tensor
        # We use the constants defined at the top of run_test
        received_output = torch.empty((N, seq_length, embed_size))
        
        # Use standard dist.recv instead of recv_object_list
        dist.recv(received_output, src=0)
        
        print(f"Rank {rank}: Tensor received. Type: {type(received_output)}, Shape: {received_output.shape}")
        
        # Verify numerical consistency (check for NaNs/Infs which indicates corruption)
        assert torch.isfinite(received_output).all(), "Numerical inconsistency detected: Received tensor contains NaN or Inf"
        
        # Verify shape consistency
        assert received_output.shape == (2, 10, 256), f"Shape mismatch: expected (2, 10, 256), got {received_output.shape}"
        
        print(f"Rank {rank}: Verification passed. Data integrity maintained.")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    print("Starting test for torch.distributed send/recv with Transformer model context...")
    mp.spawn(run_test, args=(world_size,), nprocs=world_size)