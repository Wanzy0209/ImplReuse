import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# Test case for Issue 164247: Dynamo graph break on flex attention code
# 
# The issue involves a graph break when using create_block_mask with a closure
# that captures tensors inside a torch.compile(fullgraph=True) context.
# 
# Similarity Note: The bug reproduction involves tensor reshaping operations (.view()),
# which is semantically similar to the behavior of tf.compat.v1.nn.space_to_depth.
# This test case preserves that tensor manipulation pattern.

class FlexAttentionReshapeModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(64, 64)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Process input
        processed = self.lin(x)

        # Create an intermediate tensor to be captured in the mask function
        intermediate = processed.sum(dim=-1).detach()  # Shape: (batch, seq_len)

        def dynamic_mask_function(batch_idx, head_idx, q_idx, kv_idx):
            # Capturing 'intermediate' here is the source of the graph break
            threshold = intermediate[batch_idx, q_idx % seq_len]
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

        # Reshape tensors (Pattern similar to space_to_depth/pixel_unshuffle)
        # This view operation is part of the context where the bug manifests
        q = processed.view(batch_size, 1, seq_len, self.dim)
        k = processed.view(batch_size, 1, seq_len, self.dim)
        v = processed.view(batch_size, 1, seq_len, self.dim)

        # The flex_attention call interacts with the outer torch.compile
        out = flex_attention(q, k, v, block_mask=block_mask)

        return out

def test_issue_164247():
    """
    Tests that create_block_mask works correctly with torch.compile(fullgraph=True)
    when the mask function captures intermediate tensors and involves reshaping.
    """
    model = FlexAttentionReshapeModel()
    
    # fullgraph=True is required to trigger the specific Dynamo behavior described in the issue
    compiled_model = torch.compile(model, fullgraph=True)
    
    input_tensor = torch.randn(2, 128, 64)
    
    try:
        output = compiled_model(input_tensor)
        
        # Assertions to verify correct execution
        assert output is not None, "Output should not be None"
        assert output.shape == (2, 1, 128, 64), f"Expected shape (2, 1, 128, 64), got {output.shape}"
        
        print("Test passed: No graph break detected.")
        
    except Exception as e:
        print(f"Test failed: Graph break or error occurred - {e}")
        raise

if __name__ == "__main__":
    test_issue_164247()