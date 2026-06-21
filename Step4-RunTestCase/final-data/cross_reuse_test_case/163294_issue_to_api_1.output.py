import torch
import torch.nn.functional as F
import sys

# Check for torch.export availability (requires PyTorch >= 2.1)
if not hasattr(torch, 'export'):
    print("Skipping test: torch.export is not available. Requires PyTorch >= 2.1.")
    sys.exit(0)

# Test case adapted from Issue 163294 to verify torch.nn.functional.elu_
# The original issue involved a bug where retracing (re-exporting) a module
# containing a specific operator (torch.no_grad) resulted in an empty submodule.
# This test applies the same "Export -> Re-export" stress test to the similar API.

class EluCase(torch.nn.Module):
    def forward(self, x):
        # Use the similar API: torch.nn.functional.elu_
        # This is an in-place operation, similar to how torch.no_grad affects state
        F.elu_(x, alpha=1.0)
        return x

# First export
ep = torch.export.export(
    EluCase(),
    (torch.randn(6),),
    strict=False,
)
print("First Export:")
print(ep)

# Second export (retracing) - this is where the original bug manifested
ep2 = torch.export.export(ep.module(), (torch.randn(6),))
print("\nSecond Export (Retracing):")
print(ep2)

# Verify that the re-exported program is functional and correct
input_tensor = torch.randn(6)
original_model = EluCase()
expected_output = original_model(input_tensor.clone())

# Run the re-exported module
actual_output = ep2.module()(input_tensor.clone())

# Assertion to ensure correctness is maintained after re-export
assert torch.allclose(expected_output, actual_output), "Mismatch between original and re-exported model output"