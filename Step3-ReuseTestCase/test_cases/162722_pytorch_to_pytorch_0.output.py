import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    """
    Minimal reproduction of the CausalAttention module from the bug report.
    """
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
        
        # Reshape for multi-head attention
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Einsum for energy calculation
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        # Masking
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        # Attention
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        
        # Output calculation
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out

def test_torch_compile_numerical_consistency():
    """
    Test case to verify numerical consistency between eager and compiled modes
    for the CausalAttention module.
    """
    # Configuration
    embed_size = 256
    heads = 8
    batch_size = 2
    seq_len = 10
    
    # Instantiate model
    model = CausalAttention(embed_size, heads)
    model.eval() # Set to eval mode to disable dropout (if any) for deterministic comparison
    
    # Create dummy inputs
    values = torch.randn(batch_size, seq_len, embed_size)
    keys = torch.randn(batch_size, seq_len, embed_size)
    query = torch.randn(batch_size, seq_len, embed_size)
    mask = torch.ones((batch_size, 1, 1, seq_len))
    
    # 1. Run in Eager mode
    with torch.no_grad():
        out_eager = model(values, keys, query, mask)
        
    # 2. Compile the model using torch.compile
    # This is the API under test.
    # We use default settings (inductor backend) as per the bug report context.
    compiled_model = torch.compile(model)
    
    # 3. Run in Compiled mode
    # Note: Warmup run might be needed for some backends, but usually not for correctness checks
    with torch.no_grad():
        out_compiled = compiled_model(values, keys, query, mask)
    
    # 4. Verify numerical consistency
    # The bug report mentions "severe numerical inconsistencies", so we check if outputs are close.
    # We use a tolerance that is strict enough to catch logic errors but loose enough for FP32 variance.
    is_close = torch.allclose(out_eager, out_compiled, atol=1e-4, rtol=1e-4)
    
    if not is_close:
        max_diff = (out_eager - out_compiled).abs().max().item()
        print(f"Test FAILED: Numerical inconsistency detected.")
        print(f"Max difference: {max_diff}")
        print(f"Eager output sample: {out_eager[0, 0, :5]}")
        print(f"Compiled output sample: {out_compiled[0, 0, :5]}")
        raise AssertionError(f"Compiled model output differs from eager mode by {max_diff}")
    else:
        print("Test PASSED: Compiled model output is consistent with eager mode.")

if __name__ == "__main__":
    test_torch_compile_numerical_consistency()