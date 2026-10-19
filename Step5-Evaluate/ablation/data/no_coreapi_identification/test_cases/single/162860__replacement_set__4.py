import torch


def inner(x):
    return x + 1


@torch.compile(backend="eager")
def fn(x):
    x = inner(x)
    return inner(x)


# Replaced torch.ones(3) with torch.normal(0, 1, (3,))
# to generate random input data instead of ones.
input_tensor = torch.normal(0, 1, (3,))
result = fn(input_tensor)

# Basic assertion to verify the execution
assert result.shape == (3,)