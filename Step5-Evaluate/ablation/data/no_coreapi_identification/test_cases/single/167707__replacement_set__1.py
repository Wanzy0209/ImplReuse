import torch
import os

# Setup directory and worker name for the trace handler
worker_name = "trace"
log_dir = "."

# Clean up any existing trace files from previous runs to ensure a clean test
if os.path.exists(log_dir):
    for f in os.listdir(log_dir):
        if f.startswith(worker_name) and (f.endswith(".pt.trace.json.gz") or f.endswith(".pt.trace.json")):
            os.remove(os.path.join(log_dir, f))

# Initialize the profiler with the tensorboard_trace_handler
profile = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        log_dir, worker_name=worker_name, use_gzip=True
    ),
)

profile.start()

# Original code: x = torch.randn(2, device="cuda")
# Replacement: torch.var_mean
# We create a tensor using torch.ones to serve as input for var_mean, 
# as var_mean requires an input tensor and does not generate one from scratch like randn.
x = torch.ones(2, device="cuda")
# Perform the operation using the replacement API
var, mean = torch.var_mean(x)

# Keep the original arithmetic logic to ensure workload for the profiler
y = x + 21
z = x * 15

profile.stop()

# Verify that the trace file was successfully created
files = os.listdir(log_dir)
# The handler creates a file named {worker_name}.{timestamp}.pt.trace.json.gz
trace_files = [f for f in files if f.startswith(worker_name) and f.endswith(".gz")]

assert len(trace_files) > 0, f"Trace file was not saved in {log_dir}"
print(f"Successfully saved trace file: {trace_files[0]}")