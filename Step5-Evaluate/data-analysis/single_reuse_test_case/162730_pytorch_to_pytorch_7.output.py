import torch

# Check for MPS availability
if torch.backends.mps.is_available():
    device = 'mps'
else:
    print("MPS device not found. This test requires MPS to run.")
    exit()

# Create test tensors on MPS
# We create a base tensor to rearrange into non-contiguous form
Base = torch.randn(12, 64, 768, device=device)

# Create non-contiguous anchor using standard PyTorch operations (replacing einops)
# einops.rearrange(Base, "h d m -> m (h d)") is equivalent to permuting and reshaping
anchor_noncontig = Base.permute(2, 0, 1).reshape(Base.shape[2], -1)
anchor_contig = anchor_noncontig.contiguous()

# Create positive and negative tensors
positive = torch.randn(768, 12 * 64, device=device)
negative = torch.randn(768, 12 * 64, device=device)

print(f"Anchor contiguous: {anchor_contig.is_contiguous()}")
print(f"Anchor non-contiguous: {anchor_noncontig.is_contiguous()}")

# These should be identical
result1 = torch.nn.functional.triplet_margin_with_distance_loss(anchor_noncontig, positive, negative)
result2 = torch.nn.functional.triplet_margin_with_distance_loss(anchor_contig, positive, negative)

print(f"Results match: {torch.allclose(result1, result2, atol=1e-5)}")
print(f"Max difference: {torch.abs(result1 - result2).max()}")

# Compare with CPU (works correctly)
result_cpu_noncontig = torch.nn.functional.triplet_margin_with_distance_loss(
    anchor_noncontig.cpu(), positive.cpu(), negative.cpu()
)
result_cpu_contig = torch.nn.functional.triplet_margin_with_distance_loss(
    anchor_contig.cpu(), positive.cpu(), negative.cpu()
)
print(f"CPU contiguous vs non-contiguous match: {torch.allclose(result_cpu_noncontig, result_cpu_contig, atol=1e-5)}")