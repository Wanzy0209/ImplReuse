import torch
import torch._inductor.config as config

# Enable cpp_wrapper as per the bug report conditions
config.cpp_wrapper = True

# Define a module that uses torch.prod
# The bug specifically mentions redundant H2D-D2H for non-tensor arguments.
# Here, 'dim' is a non-tensor argument passed to torch.prod.
class ProdModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward = torch.compile(self.forward)

    def forward(self, x, dim):
        return torch.prod(x, dim=dim)

prod_module = ProdModule().cuda()

with torch.profiler.profile(
    with_stack=True,
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
) as prof:
    # Run inside a DeviceContext to trigger the specific behavior
    with torch.device("cuda"):
        for i in range(10):
            x = torch.randn(i + 1, 10, 10).cuda()
            # Pass a non-tensor argument 'dim' to the compiled function
            y = prod_module(x, dim=1)
            loss = y.sum()