import torch

# Create test tensors on MPS
device = 'mps'
# Create a tensor with enough dimensions to test renorm effectively
x = torch.randn(2, 5, 10, device=device)

# Create non-contiguous input via permute (replacing einops.rearrange)
# Original einops.rearrange(x, "b d m -> m (b d)") resulted in shape (10, 10)
# We use permute to guarantee a non-contiguous tensor without external dependencies.
# Shape changes from (2, 5, 10) to (10, 2, 5).
x_noncontig = x.permute(2, 0, 1)
x_contig = x_noncontig.contiguous()

print(f"Input contiguous: {x_contig.is_contiguous()}")
print(f"Input non-contiguous: {x_noncontig.is_contiguous()}")

# Parameters for renorm
p = 2
dim = 0
maxnorm = 1.0

# These should be identical
result1 = torch.renorm(x_noncontig, p, dim, maxnorm)
result2 = torch.renorm(x_contig, p, dim, maxnorm)

print(f"Results match: {torch.allclose(result1, result2, atol=1e-5)}")
print(f"Max difference: {torch.abs(result1 - result2).max()}")

# Compare with CPU (works correctly)
result_cpu_noncontig = torch.renorm(x_noncontig.cpu(), p, dim, maxnorm)
result_cpu_contig = torch.renorm(x_contig.cpu(), p, dim, maxnorm)
print(f"CPU contiguous vs non-contiguous match: {torch.allclose(result_cpu_noncontig, result_cpu_contig, atol=1e-5)}")