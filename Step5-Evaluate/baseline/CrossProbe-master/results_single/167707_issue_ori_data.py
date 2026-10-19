# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

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

x = torch.randn(2, device="cuda")
y = x + 21
z = x * 15
profile.stop()