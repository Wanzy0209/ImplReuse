import torch

# The test requires a custom backend registered to 'privateuse1'.
# We check if such a backend is available to avoid a RuntimeError.
# This handles the environment issue where the backend is missing.
if not torch._privateuse1_backend_registered:
    print("Skipping test: privateuse1 backend is not registered.")
else:
    # Create a tensor on the custom device
    t = torch.randn(4, 4, device='privateuse1')

    # This call is expected to fail (return shape [0]) if the bug affects torch.exp2
    # similar to how it affects torch.abs.
    result = torch.exp2(t)

    # Verify the shape matches the input
    assert result.shape == t.shape, f"Expected shape {t.shape}, but got {result.shape}"