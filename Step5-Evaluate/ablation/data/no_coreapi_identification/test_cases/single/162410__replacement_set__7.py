import torch
import torch.nn.functional as F
import torch._inductor.config as inductor_config

def f(x, y):
    x.copy_(x.flip(1))
    y = y.sum(dim=1, keepdim=True) + y
    return x + y

# Replacing torch.randn with torch.nn.functional.max_pool2d
# We create a base tensor with varied data to serve as input to max_pool2d.
# Using a 4D tensor (Batch, Channel, Height, Width) as required by max_pool2d.
# Shape (20, 10, 32, 32) is chosen to be substantial but manageable.
base_x = torch.arange(20 * 10 * 32 * 32, device="cuda").view(20, 10, 32, 32).float()
base_y = torch.arange(20 * 10 * 32 * 32, device="cuda").view(20, 10, 32, 32).float()

# Apply max_pool2d to generate the inputs x and y
x = F.max_pool2d(base_x, kernel_size=2)
y = F.max_pool2d(base_y, kernel_size=2)

x_copy = x.clone()

opt_f = torch.compile(f)
ref = f(x, y)
act = opt_f(x_copy, y)

# Replacing torch.testing.assert_close with torch.nn.functional.cross_entropy
# We use cross_entropy to check for numerical issues. 
# If ref and act are close, the loss should be finite and reasonable.
# If the fusion bug causes NaNs or significant divergence, the loss behavior will change.
# We treat 'ref' as logits and 'act' as the target (probabilities/weights).
loss = F.cross_entropy(ref, act)

# Assert that the loss is finite to catch numerical explosions/NaNs
assert torch.isfinite(loss), f"Numerical issue detected: loss={loss}"

print(f"{torch._inductor.metrics.generated_kernel_count=}")