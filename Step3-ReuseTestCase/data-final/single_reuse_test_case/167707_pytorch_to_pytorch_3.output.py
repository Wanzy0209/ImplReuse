import torch
import torch.nn as nn

# Check for CUDA availability to match the original reproduction context
if not torch.cuda.is_available():
    print("CUDA is not available. Skipping test.")
    exit()

worker_name = "trace_batchnorm3d"

# Setup the profiler context as described in the bug report
profile = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        ".", worker_name=worker_name, use_gzip=True
    ),
)

profile.start()

# Adapted call site: Replace the arithmetic operations with torch.nn.BatchNorm3d
# BatchNorm3d requires 5D input (N, C, D, H, W)
num_features = 16
batch_norm = nn.BatchNorm3d(num_features).cuda()

# Create input tensor on CUDA
input_tensor = torch.randn(2, num_features, 5, 5, 5, device="cuda")

# Forward pass
output = batch_norm(input_tensor)

# Backward pass to ensure full graph execution
output.sum().backward()

profile.stop()

# Assertion to verify the layer execution
assert output.shape == input_tensor.shape