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

# Original: x = torch.randn(2, device="cuda")
# Replacement: torch.nn.functional.adaptive_max_pool2d
# We create an input tensor for the pooling operation using torch.ones
# since torch.randn is the API being replaced.
input_tensor = torch.ones(1, 1, 4, 4, device="cuda")
x = F.adaptive_max_pool2d(input_tensor, output_size=(2, 2))

y = x + 21
z = x * 15

profile.stop()