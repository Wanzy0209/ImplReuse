import torch
import torch.nn.functional as F
import os
import glob

# Clean up any existing trace files from previous runs
worker_name = "trace"
for f in glob.glob(os.path.join(".", f"{worker_name}.*.pt.trace.json*")):
    if os.path.exists(f):
        os.remove(f)

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

# Replacing torch.randn with torch.nn.functional.hinge_embedding_loss
# Original: x = torch.randn(2, device="cuda")
# To use hinge_embedding_loss, we need input and target tensors.
input_tensor = torch.tensor([[1.0, 1.0]], device="cuda")
target_tensor = torch.tensor([[1.0, 1.0]], device="cuda")

x = F.hinge_embedding_loss(input_tensor, target_tensor)

# Subsequent operations adapted to the result of the replacement API
y = x + 21
z = x * 15

profile.stop()

# Assertion to verify the trace file was created
trace_files = glob.glob(os.path.join(".", f"{worker_name}.*.pt.trace.json.gz"))
assert len(trace_files) > 0, f"Expected trace file not found for worker {worker_name}"
print(f"Test passed. Trace file created: {trace_files[0]}")