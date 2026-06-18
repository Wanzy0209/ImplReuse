import torch
import torch.nn as nn
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class FlexAttentionModule(nn.Module):
    def __init__(self, head_dim=64):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, query, key, value, block_mask=None):
        return flex_attention(query, key, value, block_mask=block_mask)

def test_flex_attention_block_mask_dynamo_export():
    """
    Test case to verify that _dynamo_graph_capture_for_export can successfully
    trace flex_attention when a block_mask is passed as a keyword argument.
    
    Relates to Issue ID: 165948
    """
    # Configuration
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64

    # Initialize inputs
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Define mask creation function
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    # Create block mask
    block_mask = create_block_mask(
        causal_mask, batch_size, num_heads, seq_len, seq_len, device="cpu"
    )

    # Instantiate model
    flex_model = FlexAttentionModule(head_dim=head_dim)
    
    flex_inputs = (query, key, value)
    flex_kwargs = {"block_mask": block_mask}

    # 1. Run eager execution to get baseline output
    eager_out = flex_model(*flex_inputs, **flex_kwargs)

    # 2. Run dynamo graph capture for export
    # This is the step that was failing in the bug report.
    with torch._dynamo.config.patch(install_free_tensors=True):
        try:
            gm = _dynamo_graph_capture_for_export(flex_model)(*flex_inputs, **flex_kwargs)
            
            # 3. Verify the captured graph runs and produces consistent output
            # Note: Depending on the exact return type of _dynamo_graph_capture_for_export,
            # gm might be a GraphModule or an ExportedProgram. We assume it is callable here.
            dynamo_out = gm(*flex_inputs, **flex_kwargs)
            
            assert torch.allclose(eager_out, dynamo_out), "Output mismatch between eager and dynamo execution"
            print("Test Passed: _dynamo_graph_capture_for_export handled block_mask correctly.")
            
        except Exception as e:
            print(f"Test Failed: Exception occurred during dynamo capture: {e}")
            raise

if __name__ == "__main__":
    test_flex_attention_block_mask_dynamo_export()