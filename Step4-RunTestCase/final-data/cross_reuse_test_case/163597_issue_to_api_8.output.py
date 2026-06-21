import math
import torch
import torch.nn.functional as F

def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> torch.Tensor:
    """
    Reference implementation of scaled dot-product attention.
    This mimics the explicit matmul operations found in similar API implementations
    (like the LSTM cell's matmul step) to ensure correctness of the fast path.
    """
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

    # Explicit matmul operation, similar to the core operation in LSTMCell
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

def test_sdpa_mps_non_contiguous():
    """
    Test case for Issue 163597: SDPA MPS regression on 2.8.0.
    
    Verifies that torch.nn.functional.scaled_dot_product_attention handles
    non-contiguous tensors correctly on the MPS device. The test leverages
    a manual implementation (similar to explicit unrolling in LSTM cells)
    to verify the results against the built-in fast path.
    """
    if not torch.backends.mps.is_available():
        print("MPS device not available. Skipping test.")
        return

    batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

    # Create tensors on MPS and transpose them to make them non-contiguous.
    # This triggers the specific regression in the fast SDPA MPS implementation.
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="mps").transpose(1, 2)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="mps").transpose(1, 2)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="mps").transpose(1, 2)

    # 1. Test Built-in SDPA on MPS (Non-contiguous)
    out_mps_builtin = F.scaled_dot_product_attention(q, k, v)

    # 2. Test Built-in SDPA on CPU (Reference)
    out_cpu_builtin = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())

    # 3. Test Manual SDPA on MPS (Reference logic)
    out_mps_manual = manual_scaled_dot_product_attention(q, k, v)

    # 4. Test Built-in SDPA on MPS (Contiguous - Control)
    out_mps_cont_builtin = F.scaled_dot_product_attention(q.contiguous(), k, v)

    # Assertions
    # The MPS output (non-contiguous) should match the CPU output.
    # We use a tolerance suitable for float32 operations across different devices.
    diff_mps_cpu = torch.norm(out_cpu_builtin - out_mps_builtin.cpu())
    print(f"MPS vs CPU (non-contiguous) Norm: {diff_mps_cpu:.6f}")
    assert torch.allclose(out_mps_builtin.cpu(), out_cpu_builtin, atol=1e-4), \
        f"MPS (non-contiguous) output differs from CPU. Norm: {diff_mps_cpu}"

    # The MPS output (non-contiguous) should match the manual implementation on MPS.
    # This verifies that the fast path produces the same result as the explicit math.
    diff_mps_manual = torch.norm(out_mps_builtin - out_mps_manual)
    print(f"MPS Built-in vs Manual (non-contiguous) Norm: {diff_mps_manual:.6f}")
    assert torch.allclose(out_mps_builtin, out_mps_manual, atol=1e-4), \
        f"MPS Built-in (non-contiguous) differs from Manual implementation. Norm: {diff_mps_manual}"

    # The contiguous version should also match (sanity check).
    diff_mps_cont = torch.norm(out_cpu_builtin - out_mps_cont_builtin.cpu())
    print(f"MPS vs CPU (contiguous) Norm: {diff_mps_cont:.6f}")
    assert torch.allclose(out_mps_cont_builtin.cpu(), out_cpu_builtin, atol=1e-4), \
        f"MPS (contiguous) output differs from CPU."

    print("All tests passed.")

if __name__ == "__main__":
    test_sdpa_mps_non_contiguous()