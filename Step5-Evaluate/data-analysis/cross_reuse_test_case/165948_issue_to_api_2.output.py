import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Handle import for _dynamo_graph_capture_for_export, fallback to torch.export if unavailable
try:
    from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
    HAS_DYNAMO_EXPORT = True
except (ImportError, ModuleNotFoundError):
    HAS_DYNAMO_EXPORT = False
    try:
        import torch.export
        HAS_TORCH_EXPORT = True
    except ImportError:
        HAS_TORCH_EXPORT = False

class FlexAttentionModule(nn.Module):
    def __init__(self, head_dim=64):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, query, key, value, block_mask=None):
        return flex_attention(query, key, value, block_mask=block_mask)

def test_flex_attention_block_mask_dynamo_export():
    """
    Test case to verify that export mechanisms can successfully
    trace flex_attention when a block_mask is passed as a keyword argument.
    
    Relates to Issue ID: 165948
    """
    if not HAS_DYNAMO_EXPORT and not HAS_TORCH_EXPORT:
        print("Test Skipped: Neither torch._dynamo.functional_export nor torch.export is available.")
        return

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

    # 2. Run export
    try:
        gm = None
        if HAS_DYNAMO_EXPORT:
            # Use the specific internal function if available
            with torch._dynamo.config.patch(install_free_tensors=True):
                gm = _dynamo_graph_capture_for_export(flex_model)(*flex_inputs, **flex_kwargs)
        elif HAS_TORCH_EXPORT:
            # Fallback to the standard torch.export API
            gm = torch.export.export(flex_model, args=flex_inputs, kwargs=flex_kwargs)
        
        # 3. Verify the captured graph runs and produces consistent output
        # Note: Depending on the exact return type, gm might be a GraphModule or an ExportedProgram. 
        # Both are callable.
        dynamo_out = gm(*flex_inputs, **flex_kwargs)
        
        assert torch.allclose(eager_out, dynamo_out), "Output mismatch between eager and dynamo execution"
        print("Test Passed: Export handled block_mask correctly.")
        
    except Exception as e:
        print(f"Test Failed: Exception occurred during export: {e}")
        raise

if __name__ == "__main__":
    test_flex_attention_block_mask_dynamo_export()