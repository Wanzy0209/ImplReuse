import torch
import torch.nn.utils as utils

B, H, W, C = 20, 2, 2, 128

x = torch.randn(B, H, W, C, requires_grad=True)

# Create a view similar to the original bug report
values_view = x.view(B, H * W, C)
values_view.retain_grad()


def log_grad(name):
    def hook(grad):
        print(f"{name} hook - shape: {grad.shape}, stride: {grad.stride()}")
        return grad

    return hook


print("forward strides", values_view.shape, values_view.stride())
values_view.register_hook(log_grad("values_view"))

# Adapted call site: replacing torch.einsum with torch.nn.utils.parameters_to_vector
# parameters_to_vector expects an iterable of tensors, so we wrap values_view in a list.
result = utils.parameters_to_vector([values_view])

# Perform backward pass
# We use torch.ones_like(result) to mimic the original backward call structure
result.backward(torch.ones_like(result))

# Assertion to verify that the gradient stride matches the input stride
# This checks if the similar API respects the gradient layout expectations
# mentioned in the bug report.
assert values_view.grad.stride() == values_view.stride(), \
    f"Stride mismatch! Expected {values_view.stride()}, got {values_view.grad.stride()}"