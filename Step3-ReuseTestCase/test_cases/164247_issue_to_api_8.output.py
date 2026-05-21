import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

def test_flex_attention_graph_break_with_captured_tensor():
    """
    Test case for Issue 164247: Dynamo graph break on flex attention code.
    
    This test reproduces the bug where torch.compile (with fullgraph=True) 
    encounters a graph break when using create_block_mask with a mask function
    that captures a tensor derived from the model input.
    
    The test leverages the semantic similarity to 'tf.nn.space_to_depth' by 
    emphasizing the reshaping/blocking operations (view/reshape) that occur 
    before the mask creation, mirroring the structural transformation pattern.
    """
    
    class FlexAttentionModel(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.dim = dim
            self.lin = torch.nn.Linear(64, 64)

        def forward(self, x):
            batch_size, seq_len, _ = x.shape

            # Process input - creates fake tensors in export's fake mode
            processed = self.lin(x)

            # Create a computation that depends on the processed tensor.
            # This mimics the data transformation pattern found in space_to_depth
            # where data is reshaped or blocked before further processing.
            intermediate = processed.sum(dim=-1).detach()  # Shape: (batch, seq_len)

            def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
                # Capturing 'intermediate' tensor in the closure.
                # This access pattern is the source of the graph break in the original issue.
                threshold = intermediate[batch_idx, q_idx % seq_len]
                return (kv_idx <= q_idx) & (threshold > 0)

            block_mask = create_block_mask(
                mask_mod=mask_mod,
                B=batch_size,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=x.device,
                _compile=False,
            )

            # Reshape inputs for attention.
            # This explicit view operation aligns with the structural transformation
            # logic of the similar API (space_to_depth).
            q = processed.view(batch_size, 1, seq_len, self.dim)
            k = processed.view(batch_size, 1, seq_len, self.dim)
            v = processed.view(batch_size, 1, seq_len, self.dim)

            # The flex_attention call. When wrapped in torch.compile(fullgraph=True),
            # this is where the graph break would occur in the buggy version.
            out = flex_attention(q, k, v, block_mask=block_mask)
            return out

    # Initialize model and input
    model = FlexAttentionModel()
    input_tensor = torch.randn(2, 128, 64)

    # Compile with fullgraph=True to trigger the specific Dynamo path mentioned in the issue
    compiled_model = torch.compile(model, fullgraph=True)

    # Run the compiled model
    try:
        output = compiled_model(input_tensor)
        
        # Assertions to verify correctness
        assert output is not None, "Output should not be None"
        assert output.shape == (2, 1, 128, 64), f"Expected shape (2, 1, 128, 64), got {output.shape}"
        
        print("Test Passed: Flex attention with captured mask tensor compiled successfully without graph break.")
        
    except Exception as e:
        print(f"Test Failed: Encountered error during execution: {e}")
        raise

if __name__ == "__main__":
    test_flex_attention_graph_break_with_captured_tensor()