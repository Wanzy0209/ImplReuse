import torch
import unittest
from torch.nn.attention.flex_attention import flex_attention

class TestFlexAttentionGQABackward(unittest.TestCase):
    """
    Test case for Issue 160074: FlexAttention backward compilation failure with GQA on NVIDIA B200.
    
    The bug manifests when:
    1. Using flex_attention with enable_gqa=True.
    2. Compiled with torch.compile using backend="inductor".
    3. Running the backward pass.
    """

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    def test_gqa_backward_inductor_compilation(self):
        # Reproducer parameters from the issue
        batch_size = 2
        q_heads = 32
        kv_heads = 8  # GQA: q_heads != kv_heads
        seq_len = 4096
        head_dim = 128
        dtype = torch.bfloat16

        # Initialize tensors with requires_grad=True to test backward pass
        q = torch.randn([batch_size, q_heads, seq_len, head_dim], 
                        dtype=dtype, device=self.device, requires_grad=True)
        k = torch.randn([batch_size, kv_heads, seq_len, head_dim], 
                        dtype=dtype, device=self.device, requires_grad=True)
        v = torch.randn([batch_size, kv_heads, seq_len, head_dim], 
                        dtype=dtype, device=self.device, requires_grad=True)

        # Compile flex_attention with the specific backend that triggers the bug
        # Note: fullgraph=True is used in the original reproducer
        compiled_flex_attention = torch.compile(
            flex_attention, 
            fullgraph=True, 
            backend="inductor"
        )

        # Run forward pass
        try:
            y = compiled_flex_attention(q, k, v, enable_gqa=True)
        except Exception as e:
            self.fail(f"Forward pass failed during compilation/execution: {e}")

        # Run backward pass
        # The bug report indicates a Triton compilation failure during the backward step
        grad_output = torch.randn_like(y)
        try:
            y.backward(grad_output)
        except Exception as e:
            # Catching the specific compilation error mentioned in the issue
            error_msg = str(e)
            if "TritonGPUHoistTMEMAlloc" in error_msg or "Failures have been detected while processing an MLIR pass pipeline" in error_msg:
                self.fail(f"Issue 160074 reproduced: Backward compilation failed with error: {e}")
            else:
                self.fail(f"Backward pass failed with unexpected error: {e}")

        # Basic assertions to ensure gradients were computed
        self.assertIsNotNone(q.grad, "Q gradient is None")
        self.assertIsNotNone(k.grad, "K gradient is None")
        self.assertIsNotNone(v.grad, "V gradient is None")
        
        # Check gradient shapes match input shapes
        self.assertEqual(q.grad.shape, q.shape)
        self.assertEqual(k.grad.shape, k.shape)
        self.assertEqual(v.grad.shape, v.shape)

if __name__ == "__main__":
    unittest.main()