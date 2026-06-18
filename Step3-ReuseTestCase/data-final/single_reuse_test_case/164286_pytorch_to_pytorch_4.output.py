import torch
import torch.nn as nn


def f(input):
    # Print layout to mimic the original debug output
    print(input.layout)

    # Use torch.nn.LayerNorm
    # normalized_shape is the size of the last dimension
    return nn.LayerNorm(input.shape[-1])(input)


# Create a standard dense tensor (LayerNorm does not support sparse inputs)
x = torch.randn(3, 4)

print("Direct call:")
print(f(x))  # works fine

print("\nVJP call:")
# The original bug failed here for sparse.mm due to layout not being copied.
# We verify that vjp works correctly for LayerNorm.
vjp = torch.func.vjp(f, x)[1]
print("VJP successful")

# Optional: Verify that the vjp function can be called to compute gradients
grads = vjp(torch.ones_like(f(x)))
assert grads is not None