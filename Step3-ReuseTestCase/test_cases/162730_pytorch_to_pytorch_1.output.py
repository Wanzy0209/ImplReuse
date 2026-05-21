import torch

# Create test tensors on MPS
device = 'mps'
W = torch.randn(12, 64, 768, device=device)

# Create non-contiguous weight via permute/reshape (replacing einops for standard library compatibility)
# Original: "h d m -> m (h d)"
w_noncontig = W.permute(2, 0, 1).reshape(768, 12 * 64)
w_contig = w_noncontig.contiguous()

print(f"Weight contiguous: {w_contig.is_contiguous()}")
print(f"Weight non-contiguous: {w_noncontig.is_contiguous()}")

# Prepare lists for nested_tensor construction
# We take slices to create a list of tensors to pass to nested_tensor
list_noncontig = [w_noncontig[i] for i in range(10)]
list_contig = [w_contig[i] for i in range(10)]

# These should be identical but might not be on MPS if the bug affects nested_tensor construction
nt1 = torch.nested.nested_tensor(list_noncontig)
nt2 = torch.nested.nested_tensor(list_contig)

# Compare by converting to padded tensors for direct comparison
result1 = nt1.to_padded_tensor(0)
result2 = nt2.to_padded_tensor(0)

print(f"Results match: {torch.allclose(result1, result2, atol=1e-5)}")
print(f"Max difference: {torch.abs(result1 - result2).max()}")

# Compare with CPU (works correctly)
W_cpu = W.cpu()
w_noncontig_cpu = W_cpu.permute(2, 0, 1).reshape(768, 12 * 64)
w_contig_cpu = w_noncontig_cpu.contiguous()
list_noncontig_cpu = [w_noncontig_cpu[i] for i in range(10)]
list_contig_cpu = [w_contig_cpu[i] for i in range(10)]

nt_cpu_noncontig = torch.nested.nested_tensor(list_noncontig_cpu)
nt_cpu_contig = torch.nested.nested_tensor(list_contig_cpu)
result_cpu_noncontig = nt_cpu_noncontig.to_padded_tensor(0)
result_cpu_contig = nt_cpu_contig.to_padded_tensor(0)

print(f"CPU contiguous vs non-contiguous match: {torch.allclose(result_cpu_noncontig, result_cpu_contig, atol=1e-5)}")