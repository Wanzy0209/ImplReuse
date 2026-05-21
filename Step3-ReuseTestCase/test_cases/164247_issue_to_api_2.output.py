import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# Reusing the code pattern from torch.distributions.constraints.independent
# which wraps a base constraint to aggregate over dimensions.
# Here we adapt this pattern to wrap a base mask function for flex_attention.
class IndependentMaskWrapper:
    """
    Wraps a base mask function, similar to how _IndependentConstraint wraps a base_constraint.
    It holds state (captured_tensor) and applies logic in __call__ akin to check().
    """
    def __init__(self, base_mask_fn, captured_tensor, seq_len):
        self.base_mask_fn = base_mask_fn
        self.captured_tensor = captured_tensor
        self.seq_len = seq_len

    def __call__(self, batch_idx, head_idx, q_idx, kv_idx):
        # Call the base mask function (similar to base_constraint.check)
        base_result = self.base_mask_fn(q_idx, kv_idx)
        
        # Apply additional logic based on the captured tensor
        # This mimics the aggregation/validation logic in the similar API
        threshold = self.captured_tensor[batch_idx, q_idx % self.seq_len]
        return base_result & (threshold > 0)


class FlexAttentionModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(64, 64)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        processed = self.lin(x)
        
        # Create a tensor to be captured by the mask function
        intermediate = processed.sum(dim=-1).detach()

        # Define a simple base mask (causal)
        def causal_mask(q_idx, kv_idx):
            return kv_idx <= q_idx

        # Use the wrapper pattern similar to torch.distributions.constraints.independent
        # to encapsulate the mask logic and the captured tensor.
        mask_wrapper = IndependentMaskWrapper(causal_mask, intermediate, seq_len)

        block_mask = create_block_mask(
            mask_mod=mask_wrapper,
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

        return flex_attention(q, k, v, block_mask=block_mask)


def test_flex_attention_graph_break():
    """
    Test case to reproduce the Dynamo graph break issue when using 
    create_block_mask with a wrapper pattern similar to 
    torch.distributions.constraints.independent.
    """
    model = FlexAttentionModel()
    
    # The issue is triggered with fullgraph=True
    compiled_model = torch.compile(model, fullgraph=True)
    
    input_tensor = torch.randn(2, 128, 64)
    
    try:
        output = compiled_model(input_tensor)
        # Check output shape to ensure execution
        assert output.shape == (2, 1, 128, 64), f"Expected shape (2, 1, 128, 64), got {output.shape}"
        print("Test Passed: No graph break detected with wrapper pattern.")
    except Exception as e:
        print(f"Test Failed: Graph break or error occurred - {e}")
        raise

if __name__ == "__main__":
    test_flex_attention_graph_break()