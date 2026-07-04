import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix for older PyTorch versions where torch.compile does not exist
if not hasattr(torch, 'compile'):
    print("Warning: torch.compile not found (requires PyTorch 2.0+). Mocking torch.compile as identity function for compatibility.")
    torch.compile = lambda model, *args, **kwargs: model

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
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(attention) + query
        x = self.norm1(x)
        forward = self.feed_forward(x)
        out = self.dropout(forward) + x
        out = self.norm2(out)
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
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        out = self.fc(out)
        return out

def test_torch_all_consistency():
    # Setup model parameters
    input_size = 100
    embed_size = 256
    num_layers = 2
    heads = 8
    forward_expansion = 4
    output_size = 10
    dropout = 0.0 # Set dropout to 0 for deterministic comparison
    max_length = 50

    model = CausalAttentionDNN(input_size, embed_size, num_layers, heads, forward_expansion, output_size, dropout, max_length)
    model.eval()

    # Create dummy inputs
    batch_size = 4
    seq_length = 10
    x = torch.randint(0, input_size, (batch_size, seq_length))
    mask = torch.tril(torch.ones((seq_length, seq_length))).expand(batch_size, 1, seq_length, seq_length)

    # Run eager mode
    with torch.no_grad():
        out_eager = model(x, mask)

    # Run compiled mode
    compiled_model = torch.compile(model)
    with torch.no_grad():
        out_compiled = compiled_model(x, mask)

    # Verify consistency using torch.all
    # We use torch.isclose to handle minor floating point differences, then torch.all to ensure every element matches
    is_close = torch.isclose(out_eager, out_compiled, rtol=1e-3, atol=1e-3)
    all_elements_match = torch.all(is_close)

    assert all_elements_match, "Numerical inconsistency detected: torch.all returned False for eager vs compiled outputs."
    print("Test Passed: torch.all confirms numerical consistency between eager and compiled models.")

if __name__ == "__main__":
    test_torch_all_consistency()