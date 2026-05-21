import torch
import torch.nn as nn

B, H, W, C = 20, 2, 2, 128

x = torch.randn(B, H, W, C, requires_grad=True)
linear = nn.Linear(C, C, bias=False)

values = linear(x)
values.retain_grad()

values_view = values.view(B, H * W, C)
values_view.retain_grad()


def log_grad(name):
    def hook(grad):
        print(f"{name} hook - shape: {grad.shape}, stride: {grad.stride()}")
        # Verify that the gradient stride matches the input stride
        expected_stride = values_view.stride()
        actual_stride = grad.stride()
        assert expected_stride == actual_stride, \
            f"Stride mismatch for {name}! Expected {expected_stride}, got {actual_stride}"
        return grad

    return hook


print("forward strides", values_view.shape, values_view.stride())
values_view.register_hook(log_grad("values_view"))

# Adaptation: Replace torch.einsum with torch.take
# Create random indices to select elements from the flattened view
indices = torch.randint(0, values_view.numel(), (B * C,))

# Perform the take operation
result = torch.take(values_view, indices)

# Perform backward pass
result.backward(torch.ones_like(result))