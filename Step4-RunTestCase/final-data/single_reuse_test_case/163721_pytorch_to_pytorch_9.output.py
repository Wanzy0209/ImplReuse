import torch
from torch import nn

# Check for MPS availability
if not torch.backends.mps.is_available():
    print("MPS device is not available. Skipping test.")
    exit(0)

mps_device = torch.device("mps")

# Wrapper over torch.unsqueeze to test the similar API on MPS
class MPSUnsqueeze(nn.Module):
    __constants__ = ["dim"]
    dim: int

    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input):
        # Replacing the custom extension call with the similar API torch.unsqueeze
        return torch.unsqueeze(input, self.dim)

    def extra_repr(self):
        return str(self.dim)

# Wrapper over the Sequential layer to test the API in a model context
class TestUnsqueezeModel(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            MPSUnsqueeze(1), # Insert dimension at index 1
            nn.Flatten(),
            nn.Linear(lin1_size * 1, output_size), # Adjust input size for flatten
        )

    def forward(self, x):
        return self.model(x)

# Instantiate and run the model on MPS to verify no segfault occurs
model = TestUnsqueezeModel().to(mps_device)
input_tensor = torch.randn(2, 784, device=mps_device)
output = model(input_tensor)

# Verify output shape and device
assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
assert output.device == mps_device, f"Expected device {mps_device}, got {output.device}"

print("Test passed: torch.unsqueeze works on MPS device without segfault.")