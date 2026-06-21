import torch
import torch.nn as nn
import sys

try:
    from torch.nn.attention.flex_attention import flex_attention
    from torch.nn.attention import SDPBackend
except ModuleNotFoundError:
    print("Test skipped: torch.nn.attention module not found. This feature requires PyTorch >= 2.5.")
    sys.exit(0)

def test_flex_attention_inductor_failure():
    """
    Test case for Issue #163300.
    Verifies that torch.compile works with flex_attention when using a custom score_mod
    that performs complex indexing on module buffers.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Test skipped: CUDA not available.")
        return

    B, N, d, H = 2, 18, 32, 4

    class FlexAttentionCPB(nn.Module):
        def __init__(self, N: int, H: int = 4):
            super().__init__()
            self.H = H
            # Initialize buffers similar to the original bug report
            # to trigger the layout conversion issue.
            self.register_buffer("idx_table", torch.randint(0, N, (N, N), device=device))
            self.register_buffer("rel_table", torch.randn(N*N, H, device=device))
            self.gamma = nn.Parameter(torch.zeros(H, device=device))

        def _score_mod(self, mu: torch.Tensor):
            mu_q, mu_k = mu.unbind(2)
            gam_sig = torch.sigmoid(self.gamma)

            def score_mod(score, b, h, q, kv):
                # The indexing operation here is critical for reproducing the bug.
                l2 = self.idx_table[q, kv]
                bias = self.rel_table[l2, h]
                w_gate = gam_sig[h] * (mu_q[b, h, q] + mu_k[b, h, kv])
                return score + w_gate * bias
            return score_mod

        def forward(self, q, k, v, mu):
            return flex_attention(q, k, v, score_mod=self._score_mod(mu))

    # Instantiate and compile the module
    mod = FlexAttentionCPB(N, H).to(device)
    mod = torch.compile(mod, dynamic=False)

    # Prepare inputs
    q = torch.randn(B, H, N, d, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mu = torch.randn(B, H, 2, N, device=device)

    # Run the compiled model
    with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = mod(q, k, v, mu)

    # Assertions
    assert out is not None, "Output is None"
    assert out.shape == (B, H, N, d), f"Output shape mismatch: {out.shape} vs {(B, H, N, d)}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"
    
    print("Test passed successfully.")

if __name__ == "__main__":
    test_flex_attention_inductor_failure()