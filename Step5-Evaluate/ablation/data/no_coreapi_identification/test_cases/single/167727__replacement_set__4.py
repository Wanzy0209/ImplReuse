import torch
import torch.nn.utils

# Helper to generate tensors using torch.unique instead of torch.rand
def get_unique_tensor(shape, device, dtype=torch.complex64):
    # Generate random data to feed into unique
    # We use torch.randn as a source since torch.rand is being replaced
    flat_size = shape[0] * shape[1]
    if dtype.is_complex:
        # Generate complex random numbers
        raw = torch.randn(flat_size, dtype=dtype, device=device)
    else:
        raw = torch.randn(flat_size, dtype=dtype, device=device)
    
    # Apply torch.unique
    unique_vals = torch.unique(raw)
    
    # Ensure we have enough elements to reshape (unique might reduce size, though unlikely for random floats)
    # For the purpose of this test, we assume unique preserves size for random floats
    # or we just slice what we have.
    return unique_vals[:flat_size].view(shape)

# Test case 1: Small tensors (Success case in original bug report)
a = get_unique_tensor((64, 300), "mps")
b = get_unique_tensor((64, 100), "mps")
c = get_unique_tensor((100, 300), "mps")

out_mps = torch.addmm(a, b, c, alpha=1.0, beta=0.5)
out_cpu = torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5)

# Replacing torch.testing.assert_close with torch.nn.utils.clip_grad_norm_
torch.nn.utils.clip_grad_norm_([out_mps, out_cpu], max_norm=1.0)

# Test case 2: Large tensors (Failure case in original bug report)
a = get_unique_tensor((64, 300), "mps")
b = get_unique_tensor((64, 10000), "mps")
c = get_unique_tensor((10000, 300), "mps")

out_mps = torch.addmm(a, b, c, alpha=1.0, beta=0.5)
out_cpu = torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5)

# Replacing torch.testing.assert_close with torch.nn.utils.clip_grad_norm_
torch.nn.utils.clip_grad_norm_([out_mps, out_cpu], max_norm=1.0)