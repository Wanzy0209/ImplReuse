import torch
from torch.distributions import constraints
from torch.distributions.constraints import dependent_property

# Define a class using the similar API (dependent_property)
class CustomDistribution:
    def __init__(self, limit):
        self.limit = limit

    @dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        # Define a constraint that depends on the tensor 'limit'
        return constraints.interval(-self.limit, self.limit)

# Create test tensors on MPS
device = 'mps'
W = torch.randn(12, 64, 768, device=device)

# Create non-contiguous tensor via as_strided (replacing einops.rearrange)
# Original: einops.rearrange(W, "h d m -> m (h d)")
# This creates a view of shape (768, 768) with strides (1, 768), which is non-contiguous.
limit_noncontig = torch.as_strided(W, size=(768, 768), stride=(1, 768))
limit_contig = limit_noncontig.contiguous()

print(f"Limit contiguous: {limit_contig.is_contiguous()}")
print(f"Limit non-contiguous: {limit_noncontig.is_contiguous()}")

# Instantiate distributions
dist_noncontig = CustomDistribution(limit_noncontig)
dist_contig = CustomDistribution(limit_contig)

# Create a value to check against the constraint
# Shape must broadcast with limit (768, 768)
x = torch.randn(1, 768, 768, device=device)

# Call the API (check the constraint)
# This replaces the original torch.nn.functional.linear call
result1 = dist_noncontig.support.check(x)
result2 = dist_contig.support.check(x)

# Verify consistency on MPS
print(f"Results match: {torch.equal(result1, result2)}")
assert torch.equal(result1, result2), "Inconsistent results between contiguous and non-contiguous tensors on MPS"

# Compare with CPU (works correctly)
result_cpu_noncontig = CustomDistribution(limit_noncontig.cpu()).support.check(x.cpu())
result_cpu_contig = CustomDistribution(limit_contig.cpu()).support.check(x.cpu())
print(f"CPU contiguous vs non-contiguous match: {torch.equal(result_cpu_noncontig, result_cpu_contig)}")