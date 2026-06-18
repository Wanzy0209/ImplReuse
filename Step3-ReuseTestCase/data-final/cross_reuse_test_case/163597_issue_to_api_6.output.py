import unittest
import torch
import torch.nn.functional as F
import math

# This test case is based on Issue 163597: SDPA MPS regression on 2.8.0
# The bug affects non-contiguous tensors on the MPS device.
# The test logic preserves the original bug reproduction script.

def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> torch.Tensor:
    """Reference implementation of scaled dot product attention."""
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class TestSDPAMPSRegression(unittest.TestCase):
    def test_sdpa_mps_non_contiguous_tensors(self):
        """
        Test that F.scaled_dot_product_attention produces correct results
        for non-contiguous tensors on the MPS device.
        
        This reproduces the regression where the fast MPS implementation
        was incorrectly handling non-contiguous memory layouts.
        """
        # Check for MPS availability (similar to checking system config/flags)
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device is not available")

        batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

        # Initialize tensors on CPU first
        q_cpu = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k_cpu = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v_cpu = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Move to MPS and transpose to create non-contiguous tensors
        # The bug specifically manifests when tensors are non-contiguous
        q_mps = q_cpu.to("mps").transpose(1, 2)
        k_mps = k_cpu.to("mps").transpose(1, 2)
        v_mps = v_cpu.to("mps").transpose(1, 2)

        # Prepare CPU reference (also transposed to match shape)
        q_ref = q_cpu.transpose(1, 2)
        k_ref = k_cpu.transpose(1, 2)
        v_ref = v_cpu.transpose(1, 2)

        # --- Run Built-in SDPA ---
        out_cpu_builtin = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
        out_mps_builtin_non_cont = F.scaled_dot_product_attention(q_mps, k_mps, v_mps)
        
        # Run with contiguous tensors as a control (workaround)
        out_mps_builtin_cont = F.scaled_dot_product_attention(
            q_mps.contiguous(), k_mps.contiguous(), v_mps.contiguous()
        )

        # --- Run Manual SDPA (Reference Logic) ---
        out_mps_manual = manual_scaled_dot_product_attention(q_mps, k_mps, v_mps)

        # --- Assertions ---
        
        # 1. Verify CPU implementation is consistent
        # (Sanity check for the test setup)
        self.assertTrue(torch.allclose(out_cpu_builtin, out_cpu_builtin))

        # 2. Verify MPS Contiguous matches CPU
        # This ensures the device itself is working correctly for standard cases
        diff_cont = torch.norm(out_cpu_builtin - out_mps_builtin_cont.cpu())
        self.assertLess(diff_cont, 1e-4, 
                        f"MPS (contiguous) differs from CPU: {diff_cont}")

        # 3. Verify MPS Non-Contiguous matches CPU
        # This is the core regression check. In the buggy version, this would fail.
        diff_non_cont = torch.norm(out_cpu_builtin - out_mps_builtin_non_cont.cpu())
        self.assertLess(diff_non_cont, 1e-4, 
                        f"MPS (non-contiguous) differs from CPU: {diff_non_cont}")

        # 4. Verify Built-in MPS matches Manual MPS on non-contiguous inputs
        # This confirms the built-in op follows the mathematical definition
        diff_manual = torch.norm(out_mps_builtin_non_cont - out_mps_manual)
        self.assertLess(diff_manual, 1e-4, 
                        f"MPS Built-in differs from Manual implementation: {diff_manual}")


if __name__ == "__main__":
    unittest.main()