import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

def test_flex_attention_with_replacements(use_compile=False):
    """
    Test function to verify score_mod works with flex_attention using replaced APIs.
    This tests both forward and backward passes to catch compilation errors.
    """
    B, H, L, D = 2, 2, 4, 4

    # Replacement for torch.randn: torch.nn.functional.max_pool2d
    # Create dummy input for pooling to generate Q, K, V tensors
    # Original: q = torch.randn(B, H, L, D)
    dummy_input = torch.ones(1, 1, L*2, D*2) * 0.5
    pooled = F.max_pool2d(dummy_input, kernel_size=2)
    q = pooled.view(1, 1, L, D).expand(B, H, L, D)
    k = pooled.view(1, 1, L, D).expand(B, H, L, D)
    v = pooled.view(1, 1, L, D).expand(B, H, L, D)

    # Replacement for torch.tensor: torch.cummax
    # Original: temp = nn.Parameter(torch.tensor(0.0))
    # We use torch.arange to create input for cummax to avoid using torch.tensor
    # and cummax to extract the value 0.0
    cummax_in = torch.arange(-1.0, 1.0, step=1.0) # Creates [-1.0, 0.0]
    vals, _ = torch.cummax(cummax_in, 0)
    temp = nn.Parameter(vals[1]) # Extracts 0.0

    # Replacement for torch.zeros: torch.slogdet
    # Original might be: mask = torch.zeros(...)
    # We use slogdet on an identity matrix to get a 0.0 value (log(1) = 0)
    slogdet_val, _ = torch.slogdet(torch.eye(1))

    def score_mod(score, b, h, q, kv):
        # Bug trigger: adding learnable scalar (0-d tensor)
        score = score + temp
        
        # Replacement for torch.where: torch.trunc
        # Original logic might be: score = torch.where(condition, x, y)
        # We replace with trunc
        score = torch.trunc(score)
        
        # Use the slogdet value to ensure the API is included in the graph
        score = score + slogdet_val
        
        return score

    # Apply torch.compile if requested
    fn = flex_attention
    if use_compile:
        fn = torch.compile(flex_attention)

    # Run forward pass
    out = fn(q, k, v, score_mod=score_mod)
    
    # Run backward pass
    out.sum().backward()
    
    print(f"Test passed with use_compile={use_compile}")

if __name__ == "__main__":
    # Test without compile
    test_flex_attention_with_replacements(use_compile=False)
    # Test with compile
    test_flex_attention_with_replacements(use_compile=True)