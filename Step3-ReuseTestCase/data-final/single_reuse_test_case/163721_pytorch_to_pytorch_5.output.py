import torch
import torch.nn as nn
from torch.amp import custom_fwd, custom_bwd

# Ensure MPS is available
assert torch.backends.mps.is_available()
mps_device = torch.device("mps")

# Define a custom autograd function using torch.amp.custom_fwd
# This adapts the original 'MPSSoftshrink' which called an external library,
# to use the similar API 'custom_fwd' for handling autocast.
class CustomMPSSoftshrinkFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="mps")
    def forward(ctx, input, lambd):
        ctx.save_for_backward(input)
        ctx.lambd = lambd
        # Implement soft shrink logic using standard ops to simulate the custom kernel
        # Logic: f(x) = x - lambda if x > lambda, x + lambda if x < -lambda, 0 otherwise
        return torch.where(
            input > lambd, input - lambd,
            torch.where(input < -lambd, input + lambd, torch.zeros_like(input))
        )

    @staticmethod
    @custom_bwd(device_type="mps")
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        lambd = ctx.lambd
        # Gradient is 1 where |x| > lambda, 0 otherwise
        mask = (input.abs() > lambd).float()
        return grad_output * mask, None

class MPSSoftshrink(nn.Module):
    __constants__ = ["lambd"]
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, input):
        return CustomMPSSoftshrinkFunction.apply(input, self.lambd)

# Test the module with autocast enabled to verify custom_fwd behavior
model = MPSSoftshrink().to(mps_device)
input_tensor = torch.randn(10, 10, device=mps_device)

# Run with autocast to ensure the custom_fwd decorator is exercised
with torch.autocast(device_type="mps"):
    output = model(input_tensor)

# Basic assertions to verify execution
assert output is not None
assert output.device == mps_device
assert output.shape == input_tensor.shape
print("Test passed: torch.amp.custom_fwd executed successfully on MPS.")