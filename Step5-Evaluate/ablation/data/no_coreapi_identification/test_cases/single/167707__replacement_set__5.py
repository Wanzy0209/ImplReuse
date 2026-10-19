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

# Replaced torch.randn with torch.nn.functional.adaptive_max_pool3d
# Create input tensor for the pooling operation since torch.randn is replaced
input_data = torch.ones(1, 1, 2, 2, 2, device="cuda")
x = torch.nn.functional.adaptive_max_pool3d(input_data, output_size=(1, 1, 1))

y = x + 21
z = x * 15
profile.stop()