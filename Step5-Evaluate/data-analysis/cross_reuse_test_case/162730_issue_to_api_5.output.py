import torch
import unittest

class TestMPSLinearNonContiguous(unittest.TestCase):
    def test_linear_non_contiguous_weight(self):
        """
        Test case for Issue #162730: MPS F.Linear producing inconsistent results 
        between contiguous and non-contiguous tensors.
        
        This test leverages patterns from the similar API 
        'tf.tpu.experimental.initialize_tpu_system', specifically:
        1. Ensuring backend initialization (is_available).
        2. Explicit synchronization (async_wait -> synchronize).
        3. Cache management (_clear_caches -> empty_cache).
        """
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available, skipping test")

        # Pattern from similar API: Ensure initialization and clear caches
        # tf.tpu.experimental.initialize_tpu_system calls context.context()._clear_caches()
        torch.mps.empty_cache()

        # Create test tensors
        # Dimensions from the original bug report
        W = torch.randn(12, 64, 768, device='mps')
        x = torch.randn(1, 3, 768, device='mps')
        bias = torch.randn(768, device='mps')

        # Create non-contiguous weight via rearrange logic
        # Original: einops.rearrange(W, "h d m -> m (h d)")
        # PyTorch equivalent: permute then reshape
        # W shape: (12, 64, 768) -> permute(2, 0, 1) -> (768, 12, 64) -> reshape -> (768, 768)
        w_permuted = W.permute(2, 0, 1)
        w_noncontig = w_permuted.reshape(768, 768)
        
        # Ensure we are actually testing the non-contiguous path
        # If reshape optimized to a contiguous copy, force a non-contiguous view
        if w_noncontig.is_contiguous():
            w_noncontig = w_noncontig[:, :]

        self.assertFalse(w_noncontig.is_contiguous(), "Weight tensor must be non-contiguous for this test")

        w_contig = w_noncontig.contiguous()

        # Run linear operations
        # Pattern from similar API: context.async_wait() -> torch.mps.synchronize()
        result_noncontig = torch.nn.functional.linear(x, w_noncontig, bias)
        torch.mps.synchronize()

        result_contig = torch.nn.functional.linear(x, w_contig, bias)
        torch.mps.synchronize()

        # Assert results match
        # The bug is that these do NOT match on MPS. 
        # This assertion should fail if the bug is present.
        self.assertTrue(
            torch.allclose(result_noncontig, result_contig, atol=1e-5),
            f"MPS Linear results differ between contiguous and non-contiguous weights. "
            f"Max difference: {torch.abs(result_noncontig - result_contig).max()}"
        )

if __name__ == '__main__':
    unittest.main()