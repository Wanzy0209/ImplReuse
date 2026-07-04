import torch

# Create test tensors on MPS
device = 'mps'
W = torch.randn(12, 64, 768, device=device)

# Create non-contiguous weight via permute and reshape (replacing einops.rearrange)
# einops.rearrange(W, "h d m -> m (h d)") is equivalent to permuting dimensions
# to (m, h, d) and then reshaping to (m, h*d).
w_noncontig = W.permute(2, 0, 1).reshape(768, -1)
w_contig = w_noncontig.contiguous()

print(f"Weight contiguous: {w_contig.is_contiguous()}")
print(f"Weight non-contiguous: {w_noncontig.is_contiguous()}")

# Test torch.randn_like
# We use manual seed to ensure the generated random numbers are comparable
torch.manual_seed(0)
result1 = torch.randn_like(w_noncontig)

torch.manual_seed(0)
result2 = torch.randn_like(w_contig)

# Check if values match (they should, regardless of input contiguity)
print(f"Results match (values): {torch.allclose(result1, result2)}")

# Check if the output layout matches the input layout
print(f"Result1 (from non-contig) is_contiguous: {result1.is_contiguous()}")
print(f"Result2 (from contig) is_contiguous: {result2.is_contiguous()}")

# Compare with CPU
torch.manual_seed(0)
result_cpu_noncontig = torch.randn_like(w_noncontig.cpu())

torch.manual_seed(0)
result_cpu_contig = torch.randn_like(w_contig.cpu())

print(f"CPU Results match (values): {torch.allclose(result_cpu_noncontig, result_cpu_contig)}")