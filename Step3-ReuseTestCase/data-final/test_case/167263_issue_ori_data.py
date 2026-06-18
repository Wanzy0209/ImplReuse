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
        return grad

    return hook


print("forward strides", values_view.shape, values_view.stride())
values_view.register_hook(log_grad("values_view"))

weights = torch.randn(B, H * W, C)
result = torch.einsum("bhc,bhc->bc", weights, values_view)

result.backward(torch.ones_like(result))