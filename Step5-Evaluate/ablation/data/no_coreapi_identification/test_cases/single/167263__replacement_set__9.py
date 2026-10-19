import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

B, H, W, C = 20, 2, 2, 128

# Replacement: torch.randn -> torch.nn.functional.hinge_embedding_loss
# We use hinge_embedding_loss with reduction='none' to generate a tensor of the required shape.
# We create dummy inputs to feed into the loss function.
dummy_x_in = torch.ones(B, H, W, C, requires_grad=True)
dummy_x_target = -torch.ones(B, H, W, C)
x = F.hinge_embedding_loss(dummy_x_in, dummy_x_target, reduction='none')

linear = nn.Linear(C, C, bias=False)

values = linear(x)
values.retain_grad()

values_view = values.view(B, H * W, C)
values_view.retain_grad()

captured_grad = None

def log_grad(name):
    def hook(grad):
        global captured_grad
        print(f"{name} hook - shape: {grad.shape}, stride: {grad.stride()}")
        captured_grad = grad
        return grad

    return hook

print("forward strides", values_view.shape, values_view.stride())
values_view.register_hook(log_grad("values_view"))

# Replacement: torch.randn -> torch.nn.functional.hinge_embedding_loss
# Generating weights using the same replacement logic.
dummy_w_in = torch.ones(B, H * W, C)
dummy_w_target = -torch.ones(B, H * W, C)
weights = F.hinge_embedding_loss(dummy_w_in, dummy_w_target, reduction='none')

result = torch.einsum("bhc,bhc->bc", weights, values_view)

# Replacement: torch.ones_like -> tf.experimental.numpy.floor_divide
# We use floor_divide(result, result) to create a tensor of ones.
# We convert the PyTorch tensor to numpy for the TF function, then back to PyTorch.
# Note: This assumes result is on CPU.
grad_input = tnp.floor_divide(result.numpy(), result.numpy())
grad_tensor = torch.from_numpy(grad_input)

result.backward(grad_tensor)

# Assertion to verify the bug (stride mismatch)
# The expected behavior is that the gradient stride matches the activation stride.
assert captured_grad is not None, "Gradient was not captured"
assert captured_grad.stride() == values_view.stride(), \
    f"Stride mismatch: forward stride {values_view.stride()} vs backward stride {captured_grad.stride()}"