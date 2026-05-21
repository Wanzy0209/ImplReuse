import torch


def inner(x):
    # Use torch.any to check if any elements are non-zero
    return torch.any(x)


@torch.compile(backend="eager")
def fn(x):
    # Call inner, then call inner again on the result
    x = inner(x)
    return inner(x)


# Test case 1: Input with non-zero values
# inner([1, 2, 3]) returns True
# inner(True) returns True
input_tensor = torch.tensor([1, 2, 3])
result = fn(input_tensor)
assert result == True

# Test case 2: Input with all zeros
# inner([0, 0, 0]) returns False
# inner(False) returns False
zero_tensor = torch.tensor([0, 0, 0])
result_zero = fn(zero_tensor)
assert result_zero == False