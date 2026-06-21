import torch
import torch.nn.functional as F

def test_sdpa_mps_non_contiguous_regression():
    """
    Test case for Issue 163597: SDPA MPS regression on 2.8.0.
    
    Verifies that F.scaled_dot_product_attention handles non-contiguous
    tensors correctly on the MPS device. The bug caused a discrepancy
    between the fast MPS implementation and the reference implementation
    when inputs were non-contiguous.
    
    This test leverages the 'keys' and 'values' tensor pattern found in
    the similar API (tf.lookup.KeyValueTensorInitializer) to structure
    the input data.
    """
    # Skip if MPS is not available
    if not torch.backends.mps.is_available():
        print("MPS device not available. Skipping test.")
        return

    # Setup dimensions
    batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

    # Initialize tensors on MPS
    # Using 'keys' and 'values' naming to reflect the pattern in the similar API
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, device="mps")
    keys = torch.randn(batch_size, seq_len, num_heads, head_dim, device="mps")
    values = torch.randn(batch_size, seq_len, num_heads, head_dim, device="mps")

    # Transpose to create non-contiguous tensors (triggering the regression)
    query = query.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    # 1. Run on MPS with non-contiguous tensors (Bug scenario)
    out_mps_non_cont = F.scaled_dot_product_attention(query, keys, values)

    # 2. Run on MPS with contiguous tensors (Workaround/Expected behavior)
    out_mps_cont = F.scaled_dot_product_attention(
        query.contiguous(), keys.contiguous(), values.contiguous()
    )

    # 3. Run on CPU for ground truth
    out_cpu = F.scaled_dot_product_attention(query.cpu(), keys.cpu(), values.cpu())

    # Assertions
    # The regression causes the non-contiguous MPS path to produce incorrect results.
    # We verify that the non-contiguous output matches the contiguous output and CPU reference.
    
    diff_mps = torch.norm(out_mps_non_cont - out_mps_cont)
    diff_cpu = torch.norm(out_mps_non_cont.cpu() - out_cpu)

    print(f"Difference (MPS Non-Cont vs MPS Cont): {diff_mps.item():.6f}")
    print(f"Difference (MPS Non-Cont vs CPU):       {diff_cpu.item():.6f}")

    assert torch.allclose(out_mps_non_cont, out_mps_cont, atol=1e-5), \
        "Regression detected: MPS Non-contiguous output differs from MPS Contiguous output."
    
    assert torch.allclose(out_mps_non_cont.cpu(), out_cpu, atol=1e-5), \
        "Regression detected: MPS Non-contiguous output differs from CPU output."

if __name__ == "__main__":
    test_sdpa_mps_non_contiguous_regression()
    print("Test passed successfully.")