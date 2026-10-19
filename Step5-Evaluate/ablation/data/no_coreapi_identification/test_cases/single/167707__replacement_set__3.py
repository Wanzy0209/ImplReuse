import torch
import torch.nn.functional as F
import os
import glob

# Define directory and worker name
worker_name = "trace"
output_dir = "."

# Clean up any existing trace files to ensure a clean test environment
pattern = os.path.join(output_dir, f"{worker_name}.*.pt.trace.json*")
for f in glob.glob(pattern):
    try:
        os.remove(f)
    except OSError:
        pass

# Initialize profiler
profile = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        output_dir, worker_name=worker_name, use_gzip=True
    ),
)

profile.start()

# Original code: x = torch.randn(2, device="cuda")
# Replaced with: torch.nn.functional.adaptive_max_pool1d
# Note: adaptive_max_pool1d requires a 3D input (N, C, L).
# We create a dummy input tensor here to apply the pooling operation.
x = F.adaptive_max_pool1d(torch.ones(1, 1, 2, device="cuda"), output_size=1)

y = x + 21
z = x * 15

profile.stop()

# Verify that the trace file was created
trace_files = glob.glob(os.path.join(output_dir, f"{worker_name}.*.pt.trace.json.gz"))
assert len(trace_files) > 0, f"Expected trace file not found matching pattern: {pattern}"
print(f"Test passed. Trace file created: {trace_files[0]}")