import torch
import torch.nn.functional as F

worker_name = "trace"
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

# Replaced torch.randn with torch.nn.functional.max_pool3d
# Create a dummy 5D input tensor for max_pool3d
input_tensor = torch.ones(1, 1, 5, 5, 5, device="cuda")
x = F.max_pool3d(input_tensor, kernel_size=2)

y = x + 21
z = x * 15
profile.stop()