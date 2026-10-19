import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

def test_flex_attention_compile_scalar():
    """
    Test function to verify score_mod works with flex_attention and torch.compile
    when using a learnable scalar parameter.
    """
    B, H, L, D = 2, 2, 4, 8

    # Create inputs using torch.empty_strided to avoid torch.randn
    # Replacing torch.randn -> torch.empty_strided (implied by avoiding source)
    q = torch.empty_strided((B, H, L, D), (H*L*D, L*D, D, 1), dtype=torch.float32)
    k = torch.empty_strided((B, H, L, D), (H*L*D, L*D, D, 1), dtype=torch.float32)
    v = torch.empty_strided((B, H, L, D), (H*L*D, L*D, D, 1), dtype=torch.float32)

    # Initialize with non-zero values to ensure gradients are meaningful
    q.fill_(0.5)
    k.fill_(0.5)
    v.fill_(0.5)

    # Original: temp = nn.Parameter(torch.tensor(0.0))
    # Replacement: torch.tensor -> torch.empty_strided
    temp = nn.Parameter(torch.empty_strided((), ()))
    temp.data.fill_(0.0)

    def score_mod(score, b, h, q, kv):
        # Original logic: score = score + temp
        # No torch.where or torch.zeros used here.
        score = score + temp
        return score

    # Compile flex_attention
    # The bug report indicates that with compiling, forward fails.
    compiled_flex = torch.compile(flex_attention)

    # Run forward pass
    try:
        out = compiled_flex(q, k, v, score_mod=score_mod)
        assert out.shape == (B, H, L, D), f"Output shape mismatch: {out.shape}"
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        raise

    # Run backward pass
    # The bug report mentions backward issues without compiling, 
    # and forward issues with compiling. We test backward here to ensure stability.
    try:
        out.sum().backward()
        assert temp.grad is not None, "Gradient for temp is None"
    except Exception as e:
        print(f"Backward pass failed with error: {e}")
        raise

if __name__ == "__main__":
    test_flex_attention_compile_scalar()