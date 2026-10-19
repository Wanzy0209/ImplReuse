import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

B, H, W, C = 20, 2, 2, 128

# Replacement for torch.randn(B, H, W, C, requires_grad=True)
# We use adaptive_max_pool3d to generate the tensor.
# Input: (B, 1, 1, 1, 1), Output size: (H, W, C) -> Result: (B, 1, H, W, C)
dummy_x = torch.zeros(B, 1, 1, 1, 1, requires_grad=True)
x = F.adaptive_max_pool3d(dummy_x, (H, W, C)).squeeze(1)

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

# Replacement for torch.randn(B, H * W, C)
# We use adaptive_max_pool3d to generate the tensor.
# Input: (B, 1, 1, 1, 1), Output size: (H*W, C, 1) -> Result: (B, 1, H*W, C, 1)
dummy_w = torch.zeros(B, 1, 1, 1, 1)
weights = F.adaptive_max_pool3d(dummy_w, (H * W, C, 1)).squeeze(1).squeeze(3).reshape(B, H * W, C)

result = torch.einsum("bhc,bhc->bc", weights, values_view)

# Replacement for torch.ones_like(result)
# We use tf.keras.ops.isclose to generate the gradient tensor.
# isclose(result, result) returns a tensor of True values (1.0).
result_tf = tf.convert_to_tensor(result.numpy())
grad_tf = tf.keras.ops.isclose(result_tf, result_tf)
grad_torch = torch.from_numpy(grad_tf.numpy()).float()

result.backward(grad_torch)

# Check if the gradient stride matches the input stride
# This assertion will fail if the bug is present
assert values_view.grad.stride() == values_view.stride(), \
    f"Stride mismatch: input stride {values_view.stride()}, grad stride {values_view.grad.stride()}"