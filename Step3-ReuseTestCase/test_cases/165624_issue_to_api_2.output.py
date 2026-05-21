import torch
import torch.backends.cuda
import unittest

class TestAllowFp16Bf16ReductionMathSdp(unittest.TestCase):
    def test_duplicate_invocation_idempotency(self):
        """
        Test that calling allow_fp16_bf16_reduction_math_sdp multiple times
        (simulating a potential merge mistake like in issue #165624 where 
        joint_custom_pre_pass was executed twice) does not cause errors 
        and behaves idempotently.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Simulate the duplicate execution pattern found in the bug report
        # First invocation
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        # Second invocation (simulating the merge mistake)
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

        # Verify functionality with a simple SDPA call
        q = k = v = torch.randn(1, 2, 8, 64, device='cuda', dtype=torch.float16)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        self.assertIsNotNone(out)

        # Toggle off twice
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)

        # Verify functionality still works
        out2 = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        self.assertIsNotNone(out2)

if __name__ == "__main__":
    unittest.main()