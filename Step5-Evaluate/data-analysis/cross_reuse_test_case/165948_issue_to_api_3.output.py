import torch
import unittest

# Handle the missing module gracefully by attempting to import it
# and setting it to None if it fails, allowing us to skip the test later.
try:
    from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
except (ImportError, ModuleNotFoundError):
    _dynamo_graph_capture_for_export = None

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

@unittest.skipIf(_dynamo_graph_capture_for_export is None, "torch._dynamo.functional_export is not available in this environment")
class TestFlexAttentionDynamoExport(unittest.TestCase):
    def test_flex_attention_with_block_mask_kwarg(self):
        """
        Test that _dynamo_graph_capture_for_export correctly traces 
        flex_attention when a block_mask is passed as a keyword argument.
        
        This mirrors the pattern of capturing complex stateful inputs 
        (similar to assets/tables in TF) during graph export.
        """
        # Define the module similar to the bug report
        class FlexAttentionModule(torch.nn.Module):
            def __init__(self, head_dim=64):
                super().__init__()
                self.head_dim = head_dim

            def forward(self, query, key, value, block_mask=None):
                return flex_attention(query, key, value, block_mask=block_mask)

        flex_model = FlexAttentionModule(head_dim=64)

        # Setup inputs
        batch_size = 2
        num_heads = 4
        seq_len = 128
        head_dim = 64

        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Create the block mask (complex configuration)
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask, batch_size, num_heads, seq_len, seq_len, device="cpu"
        )

        flex_inputs = (query, key, value)
        flex_kwargs = {"block_mask": block_mask}

        # Run eager execution to get baseline
        eager_out = flex_model(*flex_inputs, **flex_kwargs)

        # Run graph capture for export
        # This is the specific API under test that was failing
        with torch._dynamo.config.patch(install_free_tensors=True):
            gm = _dynamo_graph_capture_for_export(flex_model)(*flex_inputs, **flex_kwargs)

        # Verify the exported graph runs and produces correct results
        exported_out = gm(*flex_inputs, **flex_kwargs)
        
        self.assertTrue(torch.allclose(eager_out, exported_out))

if __name__ == "__main__":
    unittest.main()