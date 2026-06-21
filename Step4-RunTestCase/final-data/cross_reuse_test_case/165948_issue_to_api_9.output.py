import torch
import sys

# Handle missing torch._dynamo module gracefully
try:
    from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
except ModuleNotFoundError:
    print("Skipping test: 'torch._dynamo' module not found.")
    print("This test requires a PyTorch version with torch._dynamo support (e.g., PyTorch 2.0+).")
    sys.exit(0)

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def test_flex_attention_block_mask_kwarg_tracing():
    """
    Test that _dynamo_graph_capture_for_export can successfully trace 
    flex_attention when the block_mask is passed as a keyword argument.
    
    This test mirrors the usage pattern of tf.signal.inverse_stft where 
    a complex configuration object (window_fn) is passed as a kwarg.
    """
    
    class FlexAttentionModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, query, key, value, block_mask=None):
            # Passing block_mask as a kwarg is the critical pattern being tested,
            # analogous to window_fn in tf.signal.inverse_stft.
            return flex_attention(query, key, value, block_mask=block_mask)

    model = FlexAttentionModule()

    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64

    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    # Create the block_mask object to be passed as a kwarg
    block_mask = create_block_mask(
        causal_mask, batch_size, num_heads, seq_len, seq_len, device="cpu"
    )

    # 1. Run eager execution to get baseline
    eager_output = model(query, key, value, block_mask=block_mask)

    # 2. Attempt to capture the graph with the kwarg present
    # This is the operation that failed in the bug report.
    with torch._dynamo.config.patch(install_free_tensors=True):
        try:
            gm = _dynamo_graph_capture_for_export(model)(query, key, value, block_mask=block_mask)
        except Exception as e:
            print(f"Graph capture failed: {e}")
            raise

    # 3. Verify the captured graph runs and produces consistent results
    dynamo_output = gm(query, key, value, block_mask=block_mask)
    
    assert torch.allclose(eager_output, dynamo_output, atol=1e-4), \
        "Output from captured graph does not match eager output"
    
    print("Test passed: _dynamo_graph_capture_for_export successfully traced flex_attention with block_mask kwarg.")

if __name__ == "__main__":
    test_flex_attention_block_mask_kwarg_tracing()