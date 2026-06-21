import torch
# Removed 'import einops' dependency and replaced with native PyTorch operations

# Create test tensors on MPS
device = 'mps'
num_groups = 12
num_channels = 768  # Must be divisible by num_groups

# Input tensor (Batch, Channels, Height, Width)
x = torch.randn(1, num_channels, 12, 64, device=device)

# Create non-contiguous weight via transpose (replacing einops.rearrange)
# GroupNorm weight shape is (num_channels)
W = torch.randn(12, 64, device=device) # 12 * 64 = 768
# Transpose dimensions to create a non-contiguous view, then flatten
# einops.rearrange(W, "h d -> d h") is equivalent to W.t()
w_noncontig = W.t().reshape(-1)
w_contig = w_noncontig.contiguous()

print(f"Weight contiguous: {w_contig.is_contiguous()}")
print(f"Weight non-contiguous: {w_noncontig.is_contiguous()}")

# These should be identical but might not be on MPS if the bug exists
result1 = torch.nn.functional.group_norm(x, num_groups, weight=w_noncontig)
result2 = torch.nn.functional.group_norm(x, num_groups, weight=w_contig)

print(f"Results match: {torch.allclose(result1, result2, atol=1e-5)}")
print(f"Max difference: {torch.abs(result1 - result2).max()}")

# Compare with CPU (works correctly)
result_cpu_noncontig = torch.nn.functional.group_norm(x.cpu(), num_groups, weight=w_noncontig.cpu())
result_cpu_contig = torch.nn.functional.group_norm(x.cpu(), num_groups, weight=w_contig.cpu())
print(f"CPU contiguous vs non-contiguous match: {torch.allclose(result_cpu_noncontig, result_cpu_contig, atol=1e-5)}")