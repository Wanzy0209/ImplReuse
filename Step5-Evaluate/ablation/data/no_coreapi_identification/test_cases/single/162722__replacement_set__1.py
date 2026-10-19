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

        # Replacement: torch.einsum -> tf.experimental.numpy.isscalar
        # Original: energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # To maintain runnability, we use the result of isscalar to construct a tensor of the expected shape.
        is_scalar = tf.experimental.numpy.isscalar(queries)
        energy = torch.zeros(N, self.heads, query_len, key_len, device=queries.device, dtype=queries.dtype)
        if is_scalar:
            energy.fill_(1.0)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Replacement: torch.einsum -> tf.experimental.numpy.isscalar
        # Original: out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        is_scalar_out = tf.experimental.numpy.isscalar(attention)
        out = torch.zeros(N, query_len, self.heads, self.head_dim, device=values.device, dtype=values.dtype)
        if is_scalar_out:
            out.fill_(1.0)
            
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
        
        # Replacement: torch.arange -> torch.var
        # Original: positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        # To maintain runnability, we compute the variance of an arange tensor and expand it.
        # This changes the logic (constant positions) but keeps the code runnable.
        temp_arange = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length)
        positions = torch.var(temp_arange, dim=1, keepdim=True).expand(N, seq_length).to(x.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        out = self.fc(out)
        return out

def test_torch_compile_replacements():
    # Setup parameters
    input_size = 100
    embed_size = 64
    num_layers = 2
    heads = 4
    forward_expansion = 2
    output_size = 10
    dropout = 0.1
    max_length = 50
    batch_size = 4
    seq_len = 10

    # Instantiate model
    model = CausalAttentionDNN(input_size, embed_size, num_layers, heads, forward_expansion, output_size, dropout, max_length)
    
    # Create dummy input and mask
    x = torch.randint(0, input_size, (batch_size, seq_len))
    mask = torch.ones((batch_size, 1, 1, seq_len))
    
    # Run uncompiled model
    model.eval()
    with torch.no_grad():
        output_uncompiled = model(x, mask)
    
    # Compile model
    compiled_model = torch.compile(model)
    
    # Run compiled model
    with torch.no_grad():
        output_compiled = compiled_model(x, mask)
    
    # Check shapes (values will differ due to logic changes from replacements)
    assert output_uncompiled.shape == (batch_size, seq_len, output_size), f"Uncompiled output shape mismatch: {output_uncompiled.shape}"
    assert output_compiled.shape == (batch_size, seq_len, output_size), f"Compiled output shape mismatch: {output_compiled.shape}"
    
    # Note: We do not check numerical equality because the replacements (e.g., einsum -> isscalar)
    # fundamentally alter the mathematical operations of the model.
    # The test verifies that torch.compile runs successfully with the replaced APIs.
    print("Test passed: torch.compile executed successfully with replaced APIs.")

if __name__ == "__main__":
    test_torch_compile_replacements()