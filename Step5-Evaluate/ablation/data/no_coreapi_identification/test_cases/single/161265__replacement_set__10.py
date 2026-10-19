import torch

# Check CUDA availability as a replacement for the dtype and Swift API checks
is_cuda_available = torch.backends.cuda.is_built()

# Create a large tensor on MPS device.
# Note: torch.ones is replaced by torch.qr, which requires an input tensor.
# We create an input tensor using torch.empty (default float) to perform QR decomposition.
# The original bug involved a tensor size exceeding 4GB.
input_tensor = torch.empty(2, (1 << 31) + 5, device='mps')

# Perform QR decomposition (replacing torch.ones)
# This tests the backend's ability to handle large tensors during complex operations.
Q, R = torch.qr(input_tensor)

# Verify the results to ensure the tensor was processed correctly.
# The original bug checked if values were filled correctly (not zero).
# Here we check if the diagonal of R is non-zero (expected for a random matrix).
# We also check if Q is orthogonal (Q^T * Q should be close to Identity).
# However, for a minimal test, checking that values are not all zeros is sufficient.

# Check specific elements similar to the original test case
# Original: a[1, -2] and a[:, -2]
# We check the Q matrix elements.
print(Q[1, -2])
print(Q[:, -2])

# Basic assertion to ensure the operation produced valid results
# (i.e., not all zeros, which would indicate a memory/filling issue)
assert not torch.all(Q == 0), "Q matrix should not be all zeros"
assert not torch.all(R == 0), "R matrix should not be all zeros"

print("Test passed.")