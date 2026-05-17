import torch
import torch.nn.functional as F

# The original bug report highlights inconsistent behavior when padding results in a dimension of size 0.
# We adapt this test case for torch.nn.functional.grid_sample to verify if it handles
# grids defining empty spatial dimensions (size 0) consistently.

# Input tensor (Batch=1, Channel=1, Height=5, Width=3)
# Corresponds to torch.ones([5, 3]) in the original bug report, adjusted for 4D input.
input_tensor = torch.ones(1, 1, 5, 3)

# Grid defining output shape (Batch=1, Height=5, Width=0, 2)
# This corresponds to the result torch.Size([5, 0]) in the bug report.
# We test if grid_sample can handle a grid with a 0-sized dimension.
grid = torch.zeros(1, 5, 0, 2)

# Test case
try:
    output = F.grid_sample(input_tensor, grid, align_corners=True)
    # If successful, verify the shape matches the expected "cropped" dimension
    assert output.shape == torch.Size([1, 1, 5, 0]), f"Expected shape (1, 1, 5, 0), got {output.shape}"
    print("Test passed: grid_sample handles 0-sized dimension correctly.")
except RuntimeError as e:
    print(f"Test failed: grid_sample raised a RuntimeError for 0-sized dimension: {e}")
except Exception as e:
    print(f"Test failed with unexpected error: {e}")