import torch
import torch.nn as nn
import sys

# Handle missing internal module gracefully
try:
    from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
except (ImportError, ModuleNotFoundError):
    print("Skipping test: torch._dynamo.functional_export is not available in this environment.")
    sys.exit(0)

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def test_dynamo_export_flex_attention_with_block_mask():
    """
    Test case to verify that _dynamo_graph_capture_for_export can correctly
    trace flex_attention when a block_mask is passed as a keyword argument.
    
    This test preserves the logic from the bug report (Issue 165948) where
    a complex configuration object (block_mask) is passed to the attention
    mechanism, similar to how configuration arguments are passed in the
    referenced similar API pattern.
    """
    
    # Define the module wrapping flex_attention
    class FlexAttentionModule(nn.Module):
        def __init__(self, head_dim=64):
            super().__init__()
            self.head_dim = head_dim

        def forward(self, query, key, value, block_mask=None):
            return flex_attention(query, key, value, block_mask=block_mask)

    # Setup inputs
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64

    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Create the block_mask (configuration object)
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(
        causal_mask, batch_size, num_heads, seq_len, seq_len, device="cpu"
    )

    flex_model = FlexAttentionModule(head_dim=head_dim)
    flex_inputs = (query, key, value)
    flex_kwargs = {"block_mask": block_mask}

    # 1. Run eager execution to get baseline output
    eager_out = flex_model(*flex_inputs, **flex_kwargs)

    # 2. Capture the graph using _dynamo_graph_capture_for_export
    # The bug report indicates that this specific configuration patch is relevant
    with torch._dynamo.config.patch(install_free_tensors=True):
        gm = _dynamo_graph_capture_for_export(flex_model)(*flex_inputs, **flex_kwargs)

    # 3. Verify the graph was captured successfully
    assert gm is not None, "Graph capture failed: gm is None"

    # 4. Run the captured graph
    graph_out = gm(*flex_inputs, **flex_kwargs)

    # 5. Assert that the outputs match
    # This ensures the tracing logic correctly handled the block_mask kwarg
    torch.testing.assert_close(eager_out, graph_out, rtol=1e-3, atol=1e-3)
    print("Test passed: _dynamo_graph_capture_for_export successfully traced flex_attention with block_mask.")

if __name__ == "__main__":
    test_dynamo_export_flex_attention_with_block_mask()