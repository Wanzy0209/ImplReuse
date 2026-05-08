import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch.nn.functional as F

B, H, Q, D, KV = 1, 1, 1, 64, 1000

# Test showing error threshold: masking <=58% causes bug, >=59% is OK
print("PyTorch flex_attention bfloat16 bug demonstration:")
print("-" * 50)

for mask_amount in [580, 590]:
    mask = create_block_mask(
        lambda b, h, q, kv: kv < mask_amount,
        B, H, Q, KV, "cuda"
    )

    torch.manual_seed(0)
    q = torch.randn(B, H, Q, D, dtype=torch.bfloat16, device="cuda") * 5
    k = torch.randn(B, H, KV, D, dtype=torch.bfloat16, device="cuda") * 5
    v = torch.randn(B, H, KV, D, dtype=torch.bfloat16, device="cuda") * 5

    flex_out = flex_attention(q, k, v, block_mask=mask)
    ref_out = F.scaled_dot_product_attention(q, k[:, :, :mask_amount], v[:, :, :mask_amount])

    error = (flex_out - ref_out).abs().max() / ref_out.abs().max()
    status = "BUG!" if error > 0.01 else "OK"
    print(f"Mask {mask_amount}/{KV} ({mask_amount/KV:.0%}): Error {error:.1%} ({status})")

print("\nBug triggers when masking ≤58% of positions with bfloat16")