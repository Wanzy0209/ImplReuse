import torch
import torch._inductor.config as inductor_config

def f(x, y):
    x.copy_(x.flip(1))
    # Adapted to use torch.prod instead of torch.sum to test the similar API
    y = y.prod(dim=1, keepdim=True) + y
    return x + y

x = torch.randn(20, 1024 * 1024, device="cuda")
x_copy = x.clone()
y = torch.randn(20, 1024 * 1024, device="cuda")

opt_f = torch.compile(f)
ref = f(x, y)
act = opt_f(x_copy, y)

torch.testing.assert_close(ref, act)
# Fixed f-string syntax to be compatible with Python versions prior to 3.8
print(f"generated_kernel_count={torch._inductor.metrics.generated_kernel_count}")