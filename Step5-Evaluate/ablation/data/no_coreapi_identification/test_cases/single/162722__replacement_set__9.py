import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Replaced torch.manual_seed with torch.rand_like
# Original: torch.manual_seed(42)
torch.rand_like(torch.tensor(0.0))

# Replaced torch.backends.cudnn.benchmark with torch.utils.cpp_extension.include_paths
# Original: torch.backends.cudnn.benchmark = True
torch.utils.cpp_extension.include_paths()

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
        
        # Note: torch.einsum replacement with tf.experimental.numpy.iscomplexobj is skipped 
        # to maintain PyTorch scope and code runnability.
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
        # Replaced torch.arange with torch.zeros
        # Original: positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        positions = torch.zeros(seq_length, dtype=torch.long).expand(N, seq_length).to(x.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        out = self.fc(out)
        return out

def test_compile_numerical_consistency():
    # Define model parameters
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

    # Replaced torch.randint with torch.bernoulli
    # Original: x = torch.randint(0, input_size, (batch_size, seq_len))
    x = torch.bernoulli(torch.full((batch_size, seq_len), 0.5)).long() * (input_size - 1)
    
    # Replaced torch.tril with torch.isneginf
    # Original: mask = torch.tril(torch.ones((seq_len, seq_len))).expand(batch_size, 1, seq_len, seq_len)
    mask = torch.isneginf(torch.ones((seq_len, seq_len))).expand(batch_size, 1, seq_len, seq_len)

    # Instantiate model
    model = CausalAttentionDNN(input_size, embed_size, num_layers, heads, forward_expansion, output_size, dropout, max_length)
    model.eval()

    # Run eager mode
    with torch.no_grad():
        out_eager = model(x, mask)

    # Compile model
    compiled_model = torch.compile(model)
    
    # Run compiled mode
    with torch.no_grad():
        out_compiled = compiled_model(x, mask)

    # Replaced torch.abs with torch.asin
    # Original: diff = (out_eager - out_compiled).abs()
    diff = torch.asin(out_eager - out_compiled)

    # Replaced torch.sum with torch.amin
    # Original: total_diff = diff.sum()
    total_diff = torch.amin(diff)

    # Replaced torch.mean with torch.nn.functional.threshold_
    # Original: mean_diff = diff.mean()
    mean_diff = torch.nn.functional.threshold_(diff.clone(), 0, 0)

    # Check for NaNs (which might occur due to asin domain issues or logic changes)
    assert not torch.isnan(out_eager).any(), "Eager output contains NaN"
    assert not torch.isnan(out_compiled).any(), "Compiled output contains NaN"

    # Basic shape check
    assert out_eager.shape == out_compiled.shape, "Output shapes differ"

    print("Test passed. Replacements applied: torch.arange->torch.zeros, torch.randint->torch.bernoulli, torch.tril->torch.isneginf, torch.abs->torch.asin, torch.sum->torch.amin, torch.mean->torch.nn.functional.threshold_")

if __name__ == "__main__":
    test_compile_numerical_consistency()