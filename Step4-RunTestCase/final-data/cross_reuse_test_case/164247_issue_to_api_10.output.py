import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Handle missing module gracefully for older PyTorch versions
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:
    create_block_mask = None
    flex_attention = None

# Leveraging the similar API (tf.compat.v1.depth_to_space) by using its 
# PyTorch semantic equivalent (pixel_shuffle) within the test logic.
# This preserves the tensor manipulation pattern found in the similar API
# while testing the original API (create_block_mask).

class MixedFakeModeModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.lin = torch.nn.Linear(64, 64)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Process input first
        processed = self.lin(x)

        # Leverage similar API pattern: depth_to_space (pixel_shuffle in PyTorch)
        # This introduces a reshape operation similar to the similar API's domain
        # to ensure the tensor handling is robust during the graph capture.
        # Reshape for pixel_shuffle: (B, L, C) -> (B, L, H, W) assuming C is divisible
        processed_reshaped = processed.view(batch_size, seq_len, 8, 8)
        # Apply pixel_shuffle (depth_to_space equivalent)
        processed_shuffled = F.pixel_shuffle(processed_reshaped, 2)
        # Project back to original dimension
        processed = processed_shuffled.view(batch_size, seq_len, -1)
        processed = self.lin(processed)

        # Create some computation that depends on processed tensor
        intermediate = processed.sum(dim=-1).detach()  # Shape: (batch, seq_len)

        def dynamic_mask_function(batch_idx, head_idx, q_idx, kv_idx):
            threshold = intermediate[
                batch_idx, q_idx % seq_len
            ]  # Access the captured tensor
            return (kv_idx <= q_idx) & (threshold > 0)

        if create_block_mask is None:
            raise RuntimeError("create_block_mask is not available. PyTorch version might be too old.")

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

        # The bug report indicates issues with torch.compile wrapping flex_attention
        # or the model itself. We test the model compilation here.
        if flex_attention is None:
            raise RuntimeError("flex_attention is not available. PyTorch version might be too old.")
            
        out = flex_attention(q, k, v, block_mask=block_mask)
        return out

def test_flex_attention_graph_break():
    """
    Test case for Issue 164247: Dynamo graph break on flex attention code.
    Verifies that create_block_mask works correctly within a torch.compile
    context when the mask function captures intermediate tensors.
    """
    if flex_attention is None or create_block_mask is None:
        print("Skipping test: torch.nn.attention.flex_attention not found. Please upgrade PyTorch to >= 2.3.")
        return

    model = MixedFakeModeModel()
    
    # The bug is triggered specifically with fullgraph=True
    compiled_model = torch.compile(model, fullgraph=True)
    
    input_tensor = torch.randn(2, 128, 64)
    
    try:
        output = compiled_model(input_tensor)
        # Basic assertion to ensure execution and shape correctness
        assert output is not None
        assert output.shape == (2, 1, 128, 64), f"Expected shape (2, 1, 128, 64), got {output.shape}"
        print("Test passed: No graph break occurred.")
    except Exception as e:
        print(f"Test failed with exception: {e}")
        raise

if __name__ == "__main__":
    test_flex_attention_graph_break()