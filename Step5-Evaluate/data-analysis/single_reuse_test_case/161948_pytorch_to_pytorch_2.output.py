import torch
from torch.distributions import constraints

# List of interval bounds to test (adapted from threads_list)
bounds_list = [(0.0, 1.0), (-5.0, 5.0), (10.0, 20.0)]

# Size of the test tensors
tensor_size = (100, 100)

print("Testing torch.distributions.constraints.interval...")

for lower, upper in bounds_list:
    # Create the constraint (replacing torch.set_num_threads)
    constraint = constraints.interval(lower, upper)

    # Create random tensors (similar to original)
    # We generate values centered around the interval to ensure mixed results
    center = (lower + upper) / 2
    spread = (upper - lower) * 2
    a = torch.randn(tensor_size) * spread + center

    # Check the constraint (replacing matmul)
    result = constraint.check(a)

    # Verify the result is a boolean tensor of the correct shape
    assert result.dtype == torch.bool, "Result should be boolean"
    assert result.shape == a.shape, "Result shape mismatch"

    # Verify specific boundary conditions
    test_tensor = torch.tensor([lower - 1.0, lower, (lower + upper) / 2, upper, upper + 1.0])
    expected = torch.tensor([False, True, True, True, False])
    
    boundary_check = constraint.check(test_tensor)
    assert torch.equal(boundary_check, expected), f"Boundary check failed for interval [{lower}, {upper}]"

    print(f"Interval [{lower}, {upper}]: Passed.")