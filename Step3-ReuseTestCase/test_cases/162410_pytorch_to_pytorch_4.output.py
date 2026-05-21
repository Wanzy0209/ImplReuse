import torch
import torch._inductor.config as inductor_config

def f(x, y):
    # Keep the in-place mutation pattern from the original bug report
    x.copy_(x.flip(1))
    # Adapt the operation to use torch.all (the similar API)
    # We check if all elements in y are positive along dimension 1
    # We cast the result to float to allow addition with y
    y = torch.all(y > 0, dim=1, keepdim=True).float() + y
    return x + y

x = torch.randn(20, 1024 * 1024, device="cuda")
x_copy = x.clone()
y = torch.randn(20, 1024 * 1024, device="cuda")
opt_f = torch.compile(f)
ref = f(x, y)
act = opt_f(x_copy, y)
torch.testing.assert_close(ref, act)
print(f"{torch._inductor.metrics.generated_kernel_count=}")