import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch.export

class FlexAttentionModule(torch.nn.Module):
    def __init__(self, head_dim=64):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, query, key, value, block_mask=None):
        return flex_attention(query, key, value, block_mask=block_mask)

def test_flex_attention_with_block_mask_export():
    """
    Test case to verify if torch.export.export (similar API) can trace 
    flex_attention with block_mask kwarg, addressing the issue found 
    in _dynamo_graph_capture_for_export.
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

    # Run eager execution to establish baseline
    eager_out = flex_model(*flex_inputs, **flex_kwargs)

    # Attempt to export using the similar API pattern (torch.export.export)
    # This mirrors the usage in the 'Similar API' code snippet provided.
    try:
        exported_program = torch.export.export(
            flex_model, 
            flex_inputs, 
            flex_kwargs, 
            strict=True
        )
        
        # Run the exported graph
        exported_out = exported_program(*flex_inputs, **flex_kwargs)
        
        # Verify that the exported graph produces the same output
        assert torch.allclose(eager_out, exported_out), "Outputs differ between eager and exported execution"
        print("Test Passed: torch.export.export successfully traced flex_attention with block_mask kwarg.")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        raise

if __name__ == "__main__":
    test_flex_attention_with_block_mask_export()