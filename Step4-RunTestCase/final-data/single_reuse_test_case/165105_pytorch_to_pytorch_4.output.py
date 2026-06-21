import torch
import torch.nn.functional as F
import sys

# Fix: Check if torch._dynamo exists before trying to access it.
# torch._dynamo was introduced in PyTorch 2.0.
if not hasattr(torch, '_dynamo'):
    print("Test skipped: torch._dynamo is not available. This test requires PyTorch 2.0 or higher.")
    sys.exit(0)

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch.manual_seed(70609)

def fuzzed_program(input_tensor, upscale_factor):
    # Adapted to use torch.nn.functional.pixel_shuffle instead of torch.matmul
    # The original API was torch.matmul, we replace it with the similar API here.
    return F.pixel_shuffle(input_tensor, upscale_factor)

# Setup inputs compatible with pixel_shuffle
# The original bug report used float16 and cuda. We mimic that environment.
device = "cuda" if torch.cuda.is_available() else "cpu"

# pixel_shuffle requires input shape (N, C * r^2, H, W)
# We choose dimensions that are non-trivial, similar to the scale in the bug report.
upscale_factor = 2
batch_size = 1
channels_in = 16  # Must be divisible by upscale_factor^2 (4)
height = 10
width = 10

# Create input tensor
# Using torch.full to mimic the deterministic nature of the fuzzer's constants where possible,
# or randn for variable inputs. Here we use randn to simulate the 'arg' inputs.
input_tensor = torch.randn(batch_size, channels_in, height, width, dtype=torch.float16, device=device)

# Run Eager mode
eager_output = fuzzed_program(input_tensor, upscale_factor)

# Run Compiled mode (torch._dynamo)
compiled_fn = torch.compile(fuzzed_program)
compiled_output = compiled_fn(input_tensor, upscale_factor)

# Verify that the outputs match to catch DDE (Divergence) bugs
try:
    torch.testing.assert_close(eager_output, compiled_output)
    print("Test passed: Eager and Compiled outputs match.")
except AssertionError as e:
    print(f"Test failed: Divergence detected.\n{e}")