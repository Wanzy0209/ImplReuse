import torch
import pytest

# Handle import error for older PyTorch versions or missing modules
try:
    from torch.nn.attention.flex_attention import flex_attention
except (ImportError, ModuleNotFoundError):
    pytest.skip("torch.nn.attention.flex_attention not found. This test requires a newer version of PyTorch.", allow_module_level=True)

# Test case for Issue 160074: FlexAttention backward compilation failure with GQA
# This test verifies that torch.compile with the inductor backend can handle
# the backward pass of flex_attention when Grouped Query Attention (GQA) is enabled.
# The issue relates to graph execution/compilation failures, similar in concept
# to graph execution errors in other frameworks (e.g., tf.errors.OperatorNotAllowedInGraphError).

def test_flex_attention_gqa_backward_inductor():
    # Skip if CUDA is not available, as the issue is specific to NVIDIA GPUs
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Compile flex_attention with fullgraph and inductor backend
    # This configuration triggered the compilation failure in the original issue
    inductor = torch.compile(flex_attention, fullgraph=True, backend="inductor")

    with torch.device("cuda"):
        # Initialize tensors with GQA configuration
        # Batch size=2, Q heads=32, KV heads=8 (Grouped Query Attention), Seq len=4096, Head dim=128
        q = torch.randn([2, 32, 4096, 128], dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn([2, 8, 4096, 128], dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn([2, 8, 4096, 128], dtype=torch.bfloat16, requires_grad=True)

        # Forward pass
        y = inductor(q, k, v, enable_gqa=True)

        # Backward pass
        # The original bug occurred here during the compilation of the backward pass
        y.backward(torch.randn_like(y))

        # Assertions to verify the operation completed successfully and gradients were computed
        assert q.grad is not None, "Gradient for Q is None"
        assert k.grad is not None, "Gradient for K is None"
        assert v.grad is not None, "Gradient for V is None"
        
        # Check that gradients are not all zeros (basic sanity check)
        assert q.grad.abs().sum() > 0, "Gradient for Q is zero"
        assert k.grad.abs().sum() > 0, "Gradient for K is zero"
        assert v.grad.abs().sum() > 0, "Gradient for V is zero"

if __name__ == "__main__":
    test_flex_attention_gqa_backward_inductor()