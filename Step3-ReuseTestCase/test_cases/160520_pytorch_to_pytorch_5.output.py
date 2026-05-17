import torch
import torch._inductor.config as config

# Enable the configuration that triggers the bug
config.cpp_wrapper = True

class AnyCheckModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Compile the forward pass
        self.forward = torch.compile(self.forward)

    def forward(self, x, threshold):
        # Use torch.any with a non-tensor argument (threshold)
        # This tests if non-tensor arguments trigger redundant H2D/D2H copies
        return torch.any(x > threshold)

# Instantiate and move to CUDA
model = AnyCheckModule().cuda()

# Run under profiler and DeviceContext as per the bug report
with torch.profiler.profile(
    with_stack=True,
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
) as prof:
    with torch.device("cuda"):
        for i in range(10):
            # Create input tensor
            x = torch.randn(i, 28, 28).cuda()
            # Call with a non-tensor argument (float)
            y = model(x, 0.5)
            # Perform operation to ensure execution
            loss = y.sum()

# Verify the output is as expected (optional check for correctness)
print("Test completed successfully.")