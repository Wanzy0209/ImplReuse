import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Reshape to 5D for conv_transpose3d replacement
        # (N, L, H, D) -> (N, H, D, L, 1) to simulate channels/depth/width/height
        queries_5d = queries.permute(0, 2, 3, 1).unsqueeze(-1)
        keys_5d = keys.permute(0, 2, 3, 1).unsqueeze(-1)
        
        # Replacement: torch.einsum -> torch.nn.functional.conv_transpose3d
        # Note: This is a structural replacement as requested, though semantically different.
        # We assume keys acts as the weight here for the sake of the replacement instruction.
        energy = torch.nn.functional.conv_transpose3d(queries_5d, keys_5d, stride=1, padding=0)
        
        # Squeeze back to 4D: (N, H, D, L, 1) -> (N, L, H, D)
        # The output of conv_transpose3d will have different dimensions, we try to reshape back to expected (N, heads, query_len, key_len)
        # Original einsum output was "nhqk" -> (N, heads, query_len, key_len)
        # conv_transpose3d output is (N, C_out, D_out, H_out, W_out). 
        # Given the mismatch, we just reshape to the expected shape to allow the code to proceed.
        energy = energy.squeeze(-1).permute(0, 2, 3, 1)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        
        # Second replacement for the output einsum
        attention_5d = attention.permute(0, 2, 3, 1).unsqueeze(-1)
        values_5d = values.permute(0, 2, 3, 1).unsqueeze(-1)
        
        out = torch.nn.functional.conv_transpose3d(attention_5d, values_5d, stride=1, padding=0)
        out = out.squeeze(-1).permute(0, 2, 3, 1)
        
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
            nn.Linear(forward_expansion * embed_size, embed_size)
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
        # Replacement: torch.arange -> torch.ones
        positions = torch.ones(seq_length).expand(N, seq_length).to(x.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        out = self.fc(out)
        return out

# Test Case
if __name__ == "__main__":
    # Setup parameters
    input_size = 100
    embed_size = 256
    num_layers = 2
    heads = 8
    forward_expansion = 4
    output_size = 10
    dropout = 0.1
    max_length = 50
    batch_size = 4
    seq_len = 10

    # Instantiate model
    model = CausalAttentionDNN(input_size, embed_size, num_layers, heads, forward_expansion, output_size, dropout, max_length)
    
    # Compile model
    # Note: torch.compile is the API under test
    compiled_model = torch.compile(model)

    # Create dummy input
    x = torch.randint(0, input_size, (batch_size, seq_len))
    mask = torch.tril(torch.ones(seq_len, seq_len)).expand(batch_size, 1, seq_len, seq_len)

    # Run eager model
    model.eval()
    with torch.no_grad():
        output_eager = model(x, mask)

    # Run compiled model
    compiled_model.eval()
    with torch.no_grad():
        output_compiled = compiled_model(x, mask)

    # Check consistency
    # Note: Due to the replacement of einsum with conv_transpose3d, the numerical values will be vastly different
    # from the original logic, but we check if the compiled version matches the eager version of the MODIFIED model.
    try:
        torch.testing.assert_close(output_compiled, output_eager, rtol=1e-3, atol=1e-3)
        print("Test Passed: Compiled output matches Eager output for the modified model.")
    except AssertionError as e:
        print(f"Test Failed: Numerical inconsistency detected.\n{e}")