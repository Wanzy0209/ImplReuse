import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# Helper function mimicking the pattern of tf.compat.as_bytes
# This reuses the logic of type checking and conversion found in the similar API.
def convert_to_tensor_like(value, dtype=torch.float32):
    """
    Converts input to a tensor, similar to how tf.compat.as_bytes converts to bytes.
    This handles different input types (int, float, Tensor) to ensure compatibility
    within the dynamic mask function.
    """
    if isinstance(value, torch.Tensor):
        return value
    elif isinstance(value, (int, float)):
        return torch.tensor(value, dtype=dtype)
    else:
        raise TypeError(f"Expected binary or unicode string, got {type(value)}")


class FlexAttentionGraphBreakModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.lin = torch.nn.Linear(64, 64)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Process input - creates fake tensors in export's fake mode
        processed = self.lin(x)

        # Create computation that depends on processed tensor
        intermediate = processed.sum(dim=-1).detach()  # Shape: (batch, seq_len)

        def dynamic_mask_function(batch_idx, head_idx, q_idx, kv_idx):
            # Access the captured tensor
            raw_threshold = intermediate[batch_idx, q_idx % seq_len]
            
            # Leverage the similar API pattern (tf.compat.as_bytes) to validate/convert
            # the threshold before using it in the mask logic.
            threshold = convert_to_tensor_like(raw_threshold)
            
            return (kv_idx <= q_idx) & (threshold > 0)

        block_mask = create_block_mask(
            mask_mod=dynamic_mask_function,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=x.device,
            _compile=False,
        )
        
        q = processed.view(batch_size, 1, seq_len, self.dim)
        k = processed.view(batch_size, 1, seq_len, self.dim)
        v = processed.view(batch_size, 1, seq_len, self.dim)

        # The bug manifests when compiling flex_attention with fullgraph=True
        # while the mask function captures external tensors.
        out = flex_attention(q, k, v, block_mask=block_mask)
        return out


def test_create_block_mask_graph_break():
    """
    Test case for Issue 164247: Dynamo graph break on flex attention code.
    
    This test verifies that create_block_mask works correctly within a torch.compile
    context (fullgraph=True) when the mask function captures intermediate tensors.
    It incorporates the type-checking pattern of tf.compat.as_bytes to handle
    the captured data.
    """
    model = FlexAttentionGraphBreakModel()
    
    # Compile with fullgraph=True to trigger the potential graph break
    compiled_model = torch.compile(model, fullgraph=True)
    
    input_tensor = torch.randn(2, 128, 64)
    
    try:
        output = compiled_model(input_tensor)
        
        # Basic assertion to check execution
        assert output is not None
        assert output.shape == (2, 1, 128, 64)
        
        print("Test Passed: No graph break detected.")
        
    except Exception as e:
        print(f"Test Failed: Graph break or error occurred - {e}")
        raise


if __name__ == "__main__":
    test_create_block_mask_graph_break()