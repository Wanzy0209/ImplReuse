import torch
import torch.nn.functional as F
import unittest
import math

class TestSDPAMPSRegression(unittest.TestCase):
    """
    Test case for Issue ID: 163597 - SDPA MPS regression on 2.8.0.
    
    The bug affects non-contiguous tensors on the MPS device, specifically when 
    dispatched to the fast SDPA implementation. This test verifies that the 
    output for non-contiguous tensors matches the output for contiguous tensors
    and the CPU reference implementation.
    """

    def test_sdpa_mps_non_contiguous_tensors(self):
        # Skip if MPS is not available (e.g., running on Linux/Windows or non-Mac hardware)
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device is not available. Skipping MPS-specific test.")

        # Configuration from the bug report
        batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

        # Generate random tensors on CPU first for reproducibility
        # Shape: (Batch, Seq, Heads, Dim)
        q_cpu = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k_cpu = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v_cpu = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Move to MPS
        q_mps = q_cpu.to("mps")
        k_mps = k_cpu.to("mps")
        v_mps = v_cpu.to("mps")

        # Transpose to create non-contiguous tensors (standard attention layout)
        # Shape becomes: (Batch, Heads, Seq, Dim)
        # This operation typically makes the tensor non-contiguous in memory.
        q_mps_nc = q_mps.transpose(1, 2)
        k_mps_nc = k_mps.transpose(1, 2)
        v_mps_nc = v_mps.transpose(1, 2)

        q_cpu_nc = q_cpu.transpose(1, 2)
        k_cpu_nc = k_cpu.transpose(1, 2)
        v_cpu_nc = v_cpu.transpose(1, 2)

        # 1. Compute reference output on CPU (Ground Truth)
        # CPU implementation is expected to be correct
        out_cpu_ref = F.scaled_dot_product_attention(q_cpu_nc, k_cpu_nc, v_cpu_nc)

        # 2. Compute output on MPS with non-contiguous tensors (The Bug Case)
        out_mps_nc = F.scaled_dot_product_attention(q_mps_nc, k_mps_nc, v_mps_nc)

        # 3. Compute output on MPS with contiguous tensors (Control Group)
        # We explicitly call contiguous() to force the memory layout to be standard
        out_mps_c = F.scaled_dot_product_attention(
            q_mps_nc.contiguous(), 
            k_mps_nc.contiguous(), 
            v_mps_nc.contiguous()
        )

        # Assertions
        
        # The non-contiguous MPS result should match the CPU reference
        # atol is set to 1e-4 to account for minor floating point differences between devices
        self.assertTrue(
            torch.allclose(out_mps_nc.cpu(), out_cpu_ref, atol=1e-4),
            "MPS non-contiguous output does not match CPU reference."
        )

        # The non-contiguous MPS result should match the contiguous MPS result
        # This verifies that the fast path handles non-contiguous memory correctly
        self.assertTrue(
            torch.allclose(out_mps_nc, out_mps_c, atol=1e-4),
            "MPS non-contiguous output does not match MPS contiguous output."
        )

if __name__ == "__main__":
    unittest.main()