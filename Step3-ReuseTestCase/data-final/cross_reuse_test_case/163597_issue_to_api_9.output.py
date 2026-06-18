import torch
import torch.nn.functional as F
import math

def test_sdpa_mps_non_contiguous_regression():
    """
    Test case for Issue 163597: SDPA MPS regression on 2.8.0.
    
    The bug affects non-contiguous tensors which get dispatched to the fast 
    SDPA MPS implementation. This test verifies that the output of 
    F.scaled_dot_product_attention on MPS with non-contiguous inputs matches 
    the CPU reference implementation.
    """
    
    # Check for MPS availability
    if not torch.backends.mps.is_available():
        print("MPS device not available. Skipping test.")
        return

    # Configuration
    batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

    # Create tensors on MPS
    # We transpose to make them non-contiguous in the memory layout required by the fast path
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="mps").transpose(1, 2)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="mps").transpose(1, 2)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="mps").transpose(1, 2)

    # Verify that the query tensor is indeed non-contiguous
    assert not q.is_contiguous(), "Test setup failed: Query tensor must be non-contiguous."

    # 1. Compute result on MPS with non-contiguous tensors (Buggy Path)
    out_mps_non_cont = F.scaled_dot_product_attention(q, k, v)

    # 2. Compute result on MPS with contiguous tensors (Workaround/Expected Path)
    out_mps_cont = F.scaled_dot_product_attention(q.contiguous(), k, v)

    # 3. Compute result on CPU (Reference/Ground Truth)
    # We move the tensors to CPU to get the reference calculation
    out_cpu = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())

    # Calculate differences
    diff_non_cont = torch.norm(out_cpu - out_mps_non_cont.cpu())
    diff_cont = torch.norm(out_cpu - out_mps_cont.cpu())

    print(f"--- Test Results ---")
    print(f"MPS (non-contiguous) vs CPU Norm: {diff_non_cont.item():.6f}")
    print(f"MPS (contiguous) vs CPU Norm:     {diff_cont.item():.6f}")

    # Assertions
    # The contiguous MPS result should match CPU (sanity check)
    assert torch.allclose(out_mps_cont.cpu(), out_cpu, atol=1e-4), \
        f"Contiguous MPS path failed sanity check against CPU. Norm: {diff_cont.item()}"

    # The non-contiguous MPS result must also match CPU (Bug check)
    # If this fails, the regression is present.
    assert torch.allclose(out_mps_non_cont.cpu(), out_cpu, atol=1e-4), \
        f"Regression detected: MPS non-contiguous output differs from CPU. Norm: {diff_non_cont.item()}"

    print("Test passed: MPS non-contiguous handling is correct.")

if __name__ == "__main__":
    test_sdpa_mps_non_contiguous_regression()