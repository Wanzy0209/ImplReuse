import torch
import torch.special
import time

# Setup: Create a large tensor to mimic the large FSDP size scenario described in the bug.
# The bug report mentions 1024 tensors and fsdp_size 1024 leading to O(n^2) complexity.
# We create a (1024, 1024) tensor to test the similar API under similar load conditions.
large_tensor = torch.randn(1024, 1024)

# Execution: Call the similar API (torch.special.logsumexp)
# We measure the time to ensure it scales linearly and not quadratically like the bug.
start = time.time()
# Using dim=0 to reduce, similar to how the planner might aggregate or validate data across a dimension.
result = torch.special.logsumexp(large_tensor, dim=0)
end = time.time()

# Output: Mimic the print statement from the original test case
print(f"logsumexp cost {end - start}s")

# Assertion: Verify the result is correct
# Compare with the standard torch.logsumexp implementation to ensure the alias works as expected.
expected = torch.logsumexp(large_tensor, dim=0)
assert torch.allclose(result, expected), "torch.special.logsumexp produced incorrect results"

# Performance Assertion: Ensure it completes in reasonable time (unlike the O(n^2) bug)
# 1 second is a very generous upper bound for this operation size on modern hardware.
assert (end - start) < 1.0, "Performance regression detected: logsumexp took too long"