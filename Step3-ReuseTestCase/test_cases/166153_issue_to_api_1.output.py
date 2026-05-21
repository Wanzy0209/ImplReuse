import torch
import torch.nn.attention as attention
from torch.nn.attention.flex_attention import flex_attention

def test_flex_attention_recompile_limit():
    """
    Test case to reproduce the recompile limit issue with flex_attention.
    
    Bug Description:
    On v2.9.0, models using flex_attention with torch.compile hit the 
    recompile_limit (8) due to object ID checks on input tensors (e.g., key).
    This prevents proper fusion and degrades performance.
    
    This test creates new tensor instances in a loop to trigger the 
    ___check_obj_id guard failures mentioned in the bug report.
    """
    
    # Define a simple score_mod function required by flex_attention
    def score_mod(score, b, h, q_idx, kv_idx):
        return score

    # The function to be compiled
    def attention_fn(q, k, v):
        return flex_attention(q, k, v, score_mod=score_mod)

    # Compile the function
    # The bug report indicates that torch._dynamo hits the recompile_limit
    # when flex_attention is called with different tensor objects.
    compiled_fn = torch.compile(attention_fn)

    # Parameters
    batch_size, seq_len, n_heads, head_dim = 1, 16, 4, 8
    device = "cpu"
    dtype = torch.float32

    # Run the compiled function multiple times with new tensor instances.
    # The recompile_limit is 8, so we run 10 times to trigger the warning/error
    # described in the bug report.
    for i in range(10):
        # Create new tensors in each iteration to ensure object IDs change
        q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)

        # Execute
        out = compiled_fn(q, k, v)

        # Verify output shape
        assert out.shape == (batch_size, seq_len, n_heads, head_dim)

if __name__ == "__main__":
    test_flex_attention_recompile_limit()