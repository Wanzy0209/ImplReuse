import torch
import einops

# Create test tensors on MPS
device = 'mps'
W = torch.randn(12, 64, 768, device=device)
x = torch.randn(1, 3, 768, device=device) 
bias = torch.randn(768, device=device)

# Create non-contiguous weight via rearrange
w_noncontig = einops.rearrange(W, "h d m -> m (h d)")
w_contig = w_noncontig.contiguous()

print(f"Weight contiguous: {w_contig.is_contiguous()}")
print(f"Weight non-contiguous: {w_noncontig.is_contiguous()}")

# These should be identical but aren't on MPS
result1 = torch.nn.functional.linear(x, w_noncontig, bias)
result2 = torch.nn.functional.linear(x, w_contig, bias)

print(f"Results match: {torch.allclose(result1, result2, atol=1e-5)}")
print(f"Max difference: {torch.abs(result1 - result2).max()}")

# Compare with CPU (works correctly)
result_cpu_noncontig = torch.nn.functional.linear(x.cpu(), w_noncontig.cpu(), bias.cpu())
result_cpu_contig = torch.nn.functional.linear(x.cpu(), w_contig.cpu(), bias.cpu())
print(f"CPU contiguous vs non-contiguous match: {torch.allclose(result_cpu_noncontig, result_cpu_contig, atol=1e-5)}")