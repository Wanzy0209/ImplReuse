import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F

def f(x, y):
    x.copy_(x.flip(1))
    y = y.sum(dim=1, keepdim=True) + y
    return x + y

# Replacing torch.randn with torch.nn.functional.adaptive_max_pool3d
# We need to create input tensors for the pooling operation first.
# Using torch.ones to create dummy data for the pool input.
raw_x = torch.ones(20, 64, 16, 16, 16, device="cuda")
raw_y = torch.ones(20, 64, 16, 16, 16, device="cuda")

# Apply adaptive_max_pool3d to generate the test tensors x and y
# The output shape will be (20, 64, 8, 8, 8)
x = F.adaptive_max_pool3d(raw_x, (8, 8, 8))
x_copy = x.clone()
y = F.adaptive_max_pool3d(raw_y, (8, 8, 8))

opt_f = torch.compile(f)
ref = f(x, y)
act = opt_f(x_copy, y)

# Replacing torch.testing.assert_close with torch.testing.make_tensor
# Since make_tensor creates a tensor instead of asserting, we generate a tensor
# with the same shape as the reference to satisfy the replacement requirement.
# Note: This replacement changes the verification logic as make_tensor is a factory, not an assertion.
# To ensure the test still verifies correctness, we add a standard equality check.
torch.testing.make_tensor(ref.shape, device=ref.device, dtype=ref.dtype)
assert torch.equal(ref, act)

print(f"{torch._inductor.metrics.generated_kernel_count=}")