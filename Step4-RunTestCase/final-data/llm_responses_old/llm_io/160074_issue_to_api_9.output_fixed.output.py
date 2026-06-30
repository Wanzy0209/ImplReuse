import torch
import unittest

class TestFlexAttentionGQABackward(unittest.TestCase):
    def test_flex_attention_gqa_backward_compilation(self):
        """
        Test case for Issue #160074: FlexAttention backward compilation failure with GQA.
        
        The bug describes a failure in the Triton compilation pipeline during the 
        backward pass when using Grouped Query Attention (GQA) with the 'inductor' backend.
        """
        # Handle the case where flex_attention is not available in the installed PyTorch version
        try:
            from torch.nn.attention.flex_attention import flex_attention
        except ImportError:
            self.skipTest("torch.nn.attention.flex_attention is not available in this environment.")

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available, skipping test.")

        # Compile flex_attention with fullgraph=True and backend="inductor"
        # This configuration triggers the compilation failure described in the issue.
        compiled_flex_attention = torch.compile(
            flex_attention, fullgraph=True, backend="inductor"
        )

        with torch.device("cuda"):
            # Initialize tensors with dimensions matching the issue report.
            # Batch=2, Q_Heads=32, KV_Heads=8 (GQA ratio 4), Seq_Len=4096, Head_Dim=128
            q = torch.randn([2, 32, 4096, 128], dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn([2, 8, 4096, 128], dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn([2, 8, 4096, 128], dtype=torch.bfloat16, requires_grad=True)

            # Forward pass
            y = compiled_flex_attention(q, k, v, enable_gqa=True)

            # Backward pass
            # The issue states that Triton fails to compile the backward pass here.
            # We run this to ensure the compilation succeeds (or fails if the bug persists).
            y.backward(torch.randn_like(y))

            # Assertions to verify that gradients were computed successfully
            self.assertIsNotNone(q.grad)
            self.assertIsNotNone(k.grad)
            self.assertIsNotNone(v.grad)
            
            # Check that gradients are not all zeros (basic sanity check)
            self.assertTrue(torch.any(q.grad != 0))
            self.assertTrue(torch.any(k.grad != 0))
            self.assertTrue(torch.any(v.grad != 0))

if __name__ == "__main__":
    unittest.main()