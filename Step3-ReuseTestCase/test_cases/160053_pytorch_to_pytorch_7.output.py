import torch
import torch.special

# The original bug report highlighted an issue where torch.nn.functional.pad 
# failed for 4D inputs despite the error message claiming support.
# We adapt the test to verify that the similar API, torch.special.gammainc,
# correctly handles 4D inputs without raising dimension-related errors.

# Create 4D inputs (using ones to ensure valid positive inputs for gammainc)
a = torch.ones(2, 2, 2, 2)
b = torch.ones(2, 2, 2, 2)

# Call the similar API
result = torch.special.gammainc(a, b)

# Verify the output shape matches the input shape
assert result.shape == a.shape