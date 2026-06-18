import torch

# Adapted test case for torch.cummin based on the torch.aminmax issue
# Original issue: torch.return_types.aminmax(min=..., max=...) raises TypeError
# We verify if torch.cummin exhibits similar behavior or works correctly.

# 1. Functional Test
input_tensor = torch.tensor([1, -3, 5])
result = torch.cummin(input_tensor, dim=0)

# Verify the return type
assert isinstance(result, torch.return_types.cummin)

# Verify the values (cumulative minimum: 1, -3, -3)
expected_values = torch.tensor([1, -3, -3])
assert torch.equal(result.values, expected_values), f"Values mismatch: {result.values} vs {expected_values}"

# Verify the indices (indices of minimums: 0, 1, 1)
expected_indices = torch.tensor([0, 1, 1])
assert torch.equal(result.indices, expected_indices), f"Indices mismatch: {result.indices} vs {expected_indices}"

# 2. Return Type Construction Test (Checking for the reported issue)
# The original issue reported that constructing the return type via keyword arguments fails.
# We attempt to construct torch.return_types.cummin similarly.
try:
    # This mimics the representation shown in docs/repl
    constructed = torch.return_types.cummin(
        values=torch.tensor([1, -3, -3]),
        indices=torch.tensor([0, 1, 1])
    )
    # If this line is reached, the API supports kwargs construction
    print("torch.return_types.cummin supports keyword argument construction.")
except TypeError as e:
    # If this fails, it shares the behavior described in the bug report
    print(f"torch.return_types.cummin does not support keyword argument construction: {e}")