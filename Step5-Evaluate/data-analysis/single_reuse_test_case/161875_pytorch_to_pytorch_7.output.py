import torch
import torch.nn.functional as F

# Adapt input to 4D (Batch, Channel, Height, Width) as required by upsample_bilinear
input_data = torch.randn(1, 3, 32, 32)

# Use the extreme value from the original bug report for the size parameter
# This attempts to trigger a similar segmentation fault or memory error
# We expect a RuntimeError due to storage size overflow
try:
    output = F.upsample_bilinear(input_data, size=9223372036854775803)
    # If we reach here, the test failed because the expected error was not raised
    raise AssertionError("Test failed: Expected RuntimeError was not raised.")
except RuntimeError as e:
    # Check if the error is the expected overflow error
    if "Storage size calculation overflowed" in str(e):
        print("Test passed: Caught expected RuntimeError for storage size overflow.")
    else:
        # If it's a different RuntimeError, re-raise it
        raise