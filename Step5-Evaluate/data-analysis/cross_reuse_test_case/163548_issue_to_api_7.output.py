import torch
import torch.special
import time

# Reproduce the data scale from the bug report (1024 tensors)
# to test the similar API's performance and correctness with large inputs.
num_tensors = 1024
tensor_size = 1024

# Setup: Create a large list of tensors similar to the bug reproduction
# Original: fully_tensor = [torch.ones(1024, 1) for _ in range(1024)]
fully_tensor = [torch.ones(tensor_size, 1) for _ in range(num_tensors)]

# Action: Apply the similar API (torch.special.psi) to the data
# This leverages the similar API in the context of the original bug's data volume.
start = time.time()
results = [torch.special.psi(t) for t in fully_tensor]
end = time.time()

print(f"torch.special.psi processing cost {end - start}s")

# Assertions: Verify correctness and shape
assert len(results) == num_tensors
for res in results:
    assert res.shape == (tensor_size, 1)
    # psi(1) = -gamma (Euler-Mascheroni constant) approx -0.577
    # Since input is ones, output should be constant -0.577...
    expected = torch.tensor(-0.5772156649015329)
    assert torch.allclose(res[0], expected, atol=1e-5)