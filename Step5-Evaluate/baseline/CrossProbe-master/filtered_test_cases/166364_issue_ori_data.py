import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch.nn as nn

def test_flex_attention():
    B, H, T, D = 2, 4, 128, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    
    block_mask = create_block_mask(causal_mask, B, H, T, T, device=device)
    
    # Failing case - scalar parameter
    temp = nn.Parameter(torch.zeros((1,), dtype=dtype, device=device))
    
    def score_mod(score, b, h, q, kv):
        return score + temp  # This fails
    
    try:
        out = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
        loss = out.sum()
        loss.backward()
        print("Success with scalar parameter")
    except Exception as e:
        print(f"Failed with scalar: {e}")
    
    # Working case - batched parameter
    temp_batched = nn.Parameter(torch.zeros((B,), dtype=dtype, device=device))
    
    def score_mod_batched(score, b, h, q, kv):
        return score + temp_batched[b]  # This works
    
    try:
        out = flex_attention(q, k, v, score_mod=score_mod_batched, block_mask=block_mask)
        loss = out.sum()
        loss.backward()
        print("Success with batched parameter")
    except Exception as e:
        print(f"Failed with batched: {e}")

if __name__ == "__main__":
    test_flex_attention()