import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# Helper function adapted from the pattern of tf.compat.v1.train.assert_global_step
# This validates tensor properties (type, dtype, shape) before usage.
def assert_tensor_properties(tensor, name="tensor"):
    """Asserts basic properties of a tensor, similar to TensorFlow's assert_global_step."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor: {tensor}")
    
    # In the original TF API, it checks for integer type. Here we check for floating point
    # as 'intermediate' in the bug report is a result of sum/linear layers.
    if not tensor.dtype.is_floating_point:
        raise TypeError(f"{name} does not have floating point type: {tensor.dtype}")

    # The original API checks for scalar (ndims != 0). 
    # In this bug context, 'intermediate' is 2D (batch, seq_len), so we check ndims specifically.
    if tensor.ndim != 2:
        raise ValueError(f"{name} expected to have 2 dimensions, got {tensor.ndim}")


class MixedFakeModeModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.lin = torch.nn.Linear(64, 64)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Process input first - this creates fake tensors in export's fake mode
        processed = self.lin(x)

        # Create some computation that depends on processed tensor
        intermediate = processed.sum(dim=-1).detach()  # Shape: (batch, seq_len)

        # Leverage the similar API pattern to validate the captured tensor
        # before it is used in the dynamic mask function.
        assert_tensor_properties(intermediate, name="intermediate")

        def dynamic_mask_function(batch_idx, head_idx, q_idx, kv_idx):
            threshold = intermediate[
                batch_idx, q_idx % seq_len
            ]  # Access the captured tensor
            return (kv_idx <= q_idx) & (threshold > 0)

        block_mask = create_block_mask(
            mask_mod=dynamic_mask_function,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=x.device,
            _compile=False,  # Critical flag mentioned in the bug report
        )
        
        q = processed.view(batch_size, 1, seq_len, self.dim)
        k = processed.view(batch_size, 1, seq_len, self.dim)
        v = processed.view(batch_size, 1, seq_len, self.dim)

        # The bug report highlights a graph break issue here when using torch.compile
        out = flex_attention(q, k, v, block_mask=block_mask)

        return out


def test_flex_attention_dynamo_graph_break():
    """
    Test case for Issue 164247: Dynamo graph break on flex attention code.
    Verifies that create_block_mask works within a torch.compile context
    when the mask function captures a validated tensor.
    """
    model = MixedFakeModeModel()
    input_tensor = torch.randn(2, 128, 64)
    
    # The bug occurs specifically under torch.compile with fullgraph=True
    compiled_model = torch.compile(model, fullgraph=True)
    
    try:
        output = compiled_model(input_tensor)
        # Basic sanity check for output shape
        assert output.shape == (2, 1, 128, 64), f"Expected shape (2, 1, 128, 64), got {output.shape}"
        print("Test Passed: No graph break detected.")
    except Exception as e:
        print(f"Test Failed with exception: {e}")
        raise

if __name__ == "__main__":
    test_flex_attention_dynamo_graph_break()