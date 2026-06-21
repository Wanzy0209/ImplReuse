import torch
import torch.nn.functional as F
import math

def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    """
    Reference implementation of scaled dot product attention.
    Used to verify the correctness of the built-in implementation.
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

def test_sdpa_mps_non_contiguous_regression():
    """
    Test case for Issue 163597: SDPA MPS regression on 2.8.0.
    
    Verifies that torch.nn.functional.scaled_dot_product_attention 
    produces correct results for non-contiguous tensors on the MPS device.
    
    This test adapts the defensive rank-checking pattern observed in 
    tf.keras.metrics.sparse_top_k_categorical_accuracy to ensure 
    tensor dimensions are validated before processing.
    """
    
    # Check device availability
    if not torch.backends.mps.is_available():
        print("MPS device is not available. Skipping test.")
        return

    device = torch.device("mps")
    
    # Configuration
    batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

    # Create tensors on MPS
    # We use .transpose(1, 2) to make the tensors non-contiguous, 
    # which triggers the regression in the fast MPS implementation.
    q_mps = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device).transpose(1, 2)
    k_mps = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device).transpose(1, 2)
    v_mps = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device).transpose(1, 2)

    # Pattern from Similar API: Explicit rank/shape checking
    # tf.keras.metrics.sparse_top_k_categorical_accuracy checks y_pred_rank and y_true_rank
    # We verify the tensor ranks here to ensure the setup is correct.
    query_rank = q_mps.dim()
    key_rank = k_mps.dim()
    value_rank = v_mps.dim()
    
    assert query_rank == 4, f"Expected query rank 4, got {query_rank}"
    assert key_rank == 4, f"Expected key rank 4, got {key_rank}"
    assert value_rank == 4, f"Expected value rank 4, got {value_rank}"

    # Verify non-contiguity
    assert not q_mps.is_contiguous(), "Query tensor must be non-contiguous to reproduce the bug."

    # Create identical tensors on CPU for reference comparison
    # We copy the non-contiguous tensors to CPU. The CPU implementation 
    # handles non-contiguous tensors correctly.
    q_cpu = q_mps.cpu()
    k_cpu = k_mps.cpu()
    v_cpu = v_mps.cpu()

    # --- Run Built-in SDPA ---
    out_mps_builtin = F.scaled_dot_product_attention(q_mps, k_mps, v_mps)
    out_cpu_builtin = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)

    # --- Run Manual SDPA (Reference) ---
    # Running the manual implementation on MPS uses standard ops which are correct.
    out_mps_manual = manual_scaled_dot_product_attention(q_mps, k_mps, v_mps)

    # --- Assertions ---
    # 1. Compare MPS Built-in vs CPU Built-in
    # The regression causes MPS to diverge from the correct CPU result.
    diff_mps_cpu = torch.norm(out_cpu_builtin - out_mps_builtin.cpu())
    print(f"Difference (MPS Built-in vs CPU Built-in): {diff_mps_cpu.item():.6f}")
    
    # 2. Compare MPS Built-in vs MPS Manual
    # The manual implementation on MPS should match the built-in if the built-in is correct.
    diff_mps_manual = torch.norm(out_mps_builtin - out_mps_manual)
    print(f"Difference (MPS Built-in vs MPS Manual): {diff_mps_manual.item():.6f}")

    # Tolerance for floating point comparison
    # The bug usually results in large errors, so a standard tolerance is sufficient.
    tolerance = 1e-4
    
    assert diff_mps_cpu < tolerance, (
        f"Regression detected: MPS output differs from CPU. "
        f"This indicates the fast MPS path is failing on non-contiguous tensors. "
        f"Diff: {diff_mps_cpu.item()}"
    )
    
    assert diff_mps_manual < tolerance, (
        f"Regression detected: MPS output differs from manual reference. "
        f"Diff: {diff_mps_manual.item()}"
    )

    print("Test passed: SDPA on MPS handles non-contiguous tensors correctly.")

if __name__ == "__main__":
    test_sdpa_mps_non_contiguous_regression()