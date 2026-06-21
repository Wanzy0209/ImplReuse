import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

# Check for CUDA availability since we are testing torch.cuda.amp.custom_fwd
assert torch.cuda.is_available()
device = torch.device("cuda")

# Define a custom autograd function similar to the MPS extension in the bug report
class CustomSoftshrinkFunction(Function):
    @staticmethod
    @custom_fwd # Removed device_type='cuda' for compatibility with older PyTorch versions
    def forward(ctx, input, lambd):
        ctx.save_for_backward(input)
        ctx.lambd = lambd
        # Simple implementation of soft shrink
        return input * (input.abs() > lambd).float()

    @staticmethod
    @custom_bwd # Removed device_type='cuda' for compatibility with older PyTorch versions
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        lambd = ctx.lambd
        # Gradient for soft shrink
        mask = (input.abs() > lambd).float()
        return grad_output * mask, None

# Wrapper module
class CustomSoftshrink(nn.Module):
    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, input):
        return CustomSoftshrinkFunction.apply(input, self.lambd)

# Model definition mirroring the original bug report
class CustomSoftshrinkModel(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            CustomSoftshrink(),
            nn.Linear(lin1_size, lin2_size),
            CustomSoftshrink(),
            nn.Linear(lin2_size, lin3_size),
            CustomSoftshrink(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        return self.model(x)

# Test execution
model = CustomSoftshrinkModel().to(device)
input_tensor = torch.randn(64, 784, device=device)

# Run with autocast to ensure custom_fwd logic is exercised
# Removed device_type='cuda' for compatibility with older PyTorch versions
with torch.cuda.amp.autocast():
    output = model(input_tensor)

# Verify output shape
assert output.shape == (64, 10)
print("Test passed.")