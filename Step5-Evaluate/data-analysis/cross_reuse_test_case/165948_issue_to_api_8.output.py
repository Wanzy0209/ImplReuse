import torch
import sys

# Handle the case where torch._dynamo is not available (e.g., older PyTorch versions)
try:
    from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
except ModuleNotFoundError:
    print("Skipping test: 'torch._dynamo' module not found. Please ensure you are using a compatible PyTorch version.")
    sys.exit(0)

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class FlexAttentionModule(torch.nn.Module):
    def __init__(self, head_dim=64):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, query, key, value, block_mask=None):
        return flex_attention(query, key, value, block_mask=block_mask)

def test_flex_attention_with_block_mask_export():
    """
    Test that _dynamo_graph_capture_for_export can successfully trace
    flex_attention when a block_mask is passed as a keyword argument.
    """
    flex_model = FlexAttentionModule(head_dim=64)

    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64

    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(
        causal_mask, batch_size, num_heads, seq_len, seq_len, device="cpu"
    )

    flex_inputs = (query, key, value)
    flex_kwargs = {"block_mask": block_mask}

    # Run eager execution to get baseline
    eager_out = flex_model(*flex_inputs, **flex_kwargs)

    # Run export capture
    # This is the core of the bug report: ensuring this works with the kwarg
    with torch._dynamo.config.patch(install_free_tensors=True):
        gm = _dynamo_graph_capture_for_export(flex_model)(*flex_inputs, **flex_kwargs)

    # Verify the exported graph runs and produces the same result
    export_out = gm(*flex_inputs, **flex_kwargs)

    assert torch.allclose(eager_out, export_out, atol=1e-4), "Export output does not match eager output"

if __name__ == "__main__":
    test_flex_attention_with_block_mask_export()
    print("Test passed.")