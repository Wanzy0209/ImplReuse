import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

B, H, W, C = 20, 2, 2, 128

# Replacement for torch.randn: torch.nn.functional.max_pool3d
# We create a dummy input tensor and pass it through max_pool3d to generate the input 'x'.
# max_pool3d expects 5D input (N, C, D, H, W). We construct it to match the target shape (B, H, W, C).
x_dummy = torch.ones(B, C, 1, H, W, requires_grad=True)
x_pooled = F.max_pool3d(x_dummy, kernel_size=1)
x = x_pooled.permute(0, 3, 4, 1).contiguous()

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

# Replacement for torch.randn: torch.nn.functional.max_pool3d
# Generating 'weights' using max_pool3d.
w_dummy = torch.ones(B, C, 1, H * W, 1)
w_pooled = F.max_pool3d(w_dummy, kernel_size=1)
weights = w_pooled.permute(0, 3, 4, 1).contiguous().view(B, H * W, C)

result = torch.einsum("bhc,bhc->bc", weights, values_view)

# Replacement for torch.ones_like: tf.experimental.numpy.divmod
# To generate a tensor of ones for the backward pass, we use divmod(x, x).
# This requires converting the PyTorch tensor to a NumPy array for TF compatibility.
result_np = result.detach().cpu().numpy()
grad_tf_tuple = tf.experimental.numpy.divmod(result_np, result_np)
grad_np = grad_tf_tuple[0] # The quotient is 1
grad_torch = torch.from_numpy(grad_np).to(result.device)

result.backward(grad_torch)

# Assertion to ensure gradients were computed
assert values_view.grad is not None