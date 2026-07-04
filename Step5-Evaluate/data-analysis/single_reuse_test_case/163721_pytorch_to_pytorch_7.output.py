import torch
import torch.nn as nn
import sys

# Check for MPS availability
if not torch.backends.mps.is_available():
    print("MPS device is not available. Skipping test.")
    sys.exit(0)

mps_device = torch.device("mps")

# Context: The bug report involves a custom Softshrink kernel.
# We use the standard nn.Softshrink here to ensure the test is runnable
# without the external 'compiler' module mentioned in the bug report.
lambd = 0.5
model = nn.Softshrink(lambd=lambd)

# Prepare inputs
input_data = torch.randn(10, 10)
input_mps = input_data.to(mps_device)

# Execute on CPU and MPS
model_cpu = model
model_mps = model.to(mps_device)

output_cpu = model_cpu(input_data)
output_mps = model_mps(input_mps)

# Test torch.allclose: Verify that the MPS implementation matches the CPU implementation
# This is the core verification step adapted for the similar API.
is_close = torch.allclose(output_cpu, output_mps.cpu())

assert is_close, f"Outputs differ! Max diff: {(output_cpu - output_mps.cpu()).abs().max()}"
print("torch.allclose test passed for MPS device.")