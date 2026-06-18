import torch
import einops

def test_mps_linear_noncontiguous_weights():
    """
    Test that torch.nn.functional.linear produces consistent results
    between contiguous and non-contiguous weight tensors on MPS.
    """
    if not torch.backends.mps.is_available():
        print("MPS backend not available, skipping test.")
        return

    device = 'mps'
    
    # Create test tensors
    W = torch.randn(12, 64, 768, device=device)
    x = torch.randn(1, 3, 768, device=device) 
    bias = torch.randn(768, device=device)

    # Create non-contiguous weight via rearrange
    # This operation changes the memory layout without copying data immediately
    w_noncontig = einops.rearrange(W, "h d m -> m (h d)")
    w_contig = w_noncontig.contiguous()

    # Verify contiguity properties
    assert not w_noncontig.is_contiguous(), "w_noncontig should be non-contiguous"
    assert w_contig.is_contiguous(), "w_contig should be contiguous"

    # Compute results using the API under test
    result_noncontig = torch.nn.functional.linear(x, w_noncontig, bias)
    result_contig = torch.nn.functional.linear(x, w_contig, bias)

    # Verify that results match
    # On a fixed implementation, these should be identical.
    # The bug report indicates they differ on MPS.
    match = torch.allclose(result_noncontig, result_contig, atol=1e-5)
    max_diff = torch.abs(result_noncontig - result_contig).max()

    print(f"Results match: {match}")
    print(f"Max difference: {max_diff}")

    # Optional: Verify against CPU to ensure correctness
    with torch.no_grad():
        x_cpu = x.cpu()
        w_noncontig_cpu = w_noncontig.cpu()
        w_contig_cpu = w_contig.cpu()
        bias_cpu = bias.cpu()

        result_cpu_noncontig = torch.nn.functional.linear(x_cpu, w_noncontig_cpu, bias_cpu)
        result_cpu_contig = torch.nn.functional.linear(x_cpu, w_contig_cpu, bias_cpu)
        
        cpu_match = torch.allclose(result_cpu_noncontig, result_cpu_contig, atol=1e-5)
        print(f"CPU contiguous vs non-contiguous match: {cpu_match}")

    # Main assertion for the bug fix
    assert match, (
        f"torch.nn.functional.linear on MPS produced inconsistent results "
        f"between contiguous and non-contiguous weights. Max diff: {max_diff}"
    )

if __name__ == "__main__":
    test_mps_linear_noncontiguous_weights()