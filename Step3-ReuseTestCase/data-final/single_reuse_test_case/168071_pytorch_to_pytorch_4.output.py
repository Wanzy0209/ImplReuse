import torch
import torch.nn.functional as F

# Adapted test case for torch.nn.functional.conv2d based on Issue #168071
# Original Issue: torch.nn.functional.pad crashes when padding (0, 0) to 0-shape
# Adaptation: Verify conv2d handles 0-shape inputs correctly without crashing

# Input tensor with a 0-shape dimension (Height=0)
x0 = torch.zeros((1, 1, 0, 10))

# Weight tensor (Kernel size 1x1 to ensure output size is 0 with padding 0)
# Using kernel size 1 mimics the (0,0) padding scenario where output size remains 0
weight = torch.randn(1, 1, 1, 1)

# Call conv2d with padding (0, 0)
# Expected behavior: Output shape should be (1, 1, 0, 10)
# Bug scenario: RuntimeError regarding negative/invalid output size
try:
    x1 = F.conv2d(x0, weight, padding=(0, 0))
    print(f"Success. Output shape: {x1.shape}")
    assert x1.shape == torch.Size([1, 1, 0, 10]), f"Expected shape (1, 1, 0, 10), but got {x1.shape}"
except RuntimeError as e:
    print(f"RuntimeError encountered: {e}")