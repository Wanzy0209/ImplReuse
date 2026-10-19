import torch

# Original test case logic: Check version string validity
# Replaced with: Check backend availability
# Note: torch.version.hip is kept for context but not passed to is_available
# as it takes no arguments.

print(torch.backends.nnpack.is_available())