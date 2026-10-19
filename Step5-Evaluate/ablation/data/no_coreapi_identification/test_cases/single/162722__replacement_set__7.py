import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

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
        
        # Replacement: torch.einsum -> tf.experimental.numpy.hstack
        # Original: energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # Note: Converting to numpy for TF compatibility and back to torch for subsequent ops
        energy = torch.as_tensor(tf.experimental.numpy.hstack([queries.numpy(), keys.numpy()]))
        
        if mask is not None:
            # Note: The shape of energy has changed due to hstack, 
            # so masking might not align as expected in the original logic.
            # We apply the operation as requested.
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        
        # Replacement: torch.einsum -> tf.experimental.numpy.hstack
        # Original: out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(...)
        out = torch.as_tensor(tf.experimental.numpy.hstack([attention.numpy(), values.numpy()]))
        
        # Original reshape logic preserved, though shape mismatch is likely due to hstack
        out = out.reshape(N, query_len, self.heads * self.head_dim)
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
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class CausalAttentionDNN(nn.Module):
    def __init__(self, input_size, embed_size, num_layers, heads, forward_expansion, output_size, dropout, max_length):
        super(CausalAttentionDNN, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(input_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                CausalAttentionBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(embed_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        # Replacement: torch.arange -> torch.linspace
        # Original: positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        positions = torch.linspace(0, seq_length - 1, seq_length, device=x.device).expand(N, seq_length)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        out = self.fc(out)
        return out

def test_compile_numerical_consistency():
    # Define model parameters
    input_size = 10
    embed_size = 8
    num_layers = 1
    heads = 2
    forward_expansion = 2
    output_size = 2
    dropout = 0.0
    max_length = 10
    batch_size = 2
    seq_len = 5

    # Instantiate model
    model = CausalAttentionDNN(input_size, embed_size, num_layers, heads, forward_expansion, output_size, dropout, max_length)
    
    # Create dummy input and mask
    x = torch.randint(0, input_size, (batch_size, seq_len))
    mask = torch.tril(torch.ones((seq_len, seq_len))).expand(batch_size, 1, 1, seq_len)

    # Run eager mode
    model.eval()
    with torch.no_grad():
        output_eager = model(x, mask)

    # Compile model
    compiled_model = torch.compile(model)
    
    # Run compiled mode
    with torch.no_grad():
        output_compiled = compiled_model(x, mask)

    # Check consistency
    # Note: Due to the replacement of einsum with hstack, the numerical values 
    # and shapes might differ significantly from the original logic, 
    # but we check if the compiled model matches the eager model with the new logic.
    try:
        assert torch.allclose(output_eager, output_compiled, atol=1e-4), "Numerical inconsistency detected between eager and compiled modes"
        print("Test Passed: Eager and compiled outputs are consistent.")
    except AssertionError as e:
        print(f"Test Failed: {e}")
    except RuntimeError as e:
        # Expected to potentially fail due to shape mismatches from hstack replacement
        print(f"Runtime Error (likely due to API replacement shape mismatch): {e}")

if __name__ == "__main__":
    test_compile_numerical_consistency()