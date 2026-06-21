import torch
from torch import nn

# Fix: Handle cases where MPS is not available (e.g., non-Apple Silicon or older macOS)
# by falling back to CPU to ensure the test logic can still be verified.
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    mps_device = torch.device("cpu")
    print("Warning: MPS backend not available. Falling back to CPU for testing.")

# Wrapper over torch.diff to replace the custom MPS soft shrink kernel.
class MPSDiff(nn.Module):
    __constants__ = ["n", "axis"]
    n: int
    axis: int

    def __init__(self, n: int = 1, axis: int = -1) -> None:
        super().__init__()
        self.n = n
        self.axis = axis

    def forward(self, input):
        return torch.diff(input, n=self.n, axis=self.axis)

    def extra_repr(self):
        return f"n={self.n}, axis={self.axis}"

# Wrapper over the Sequential layer, using torch.diff.
# Note: torch.diff reduces the dimension along the specified axis by n.
# We adjust the input dimensions of subsequent Linear layers accordingly.
class CustomMPSDiffModel(nn.Module):
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
            MPSDiff(n=1, axis=-1),
            nn.Linear(lin1_size - 1, lin2_size),
            MPSDiff(n=1, axis=-1),
            nn.Linear(lin2_size - 1, lin3_size),
            MPSDiff(n=1, axis=-1),
            nn.Linear(lin3_size - 1, output_size),
        )

    def forward(self, x):
        return self.model(x)

# Test execution
if __name__ == "__main__":
    model = CustomMPSDiffModel().to(mps_device)
    input_tensor = torch.randn(1, 784, device=mps_device)
    output = model(input_tensor)
    
    # Verify output shape
    assert output.shape == (1, 10), f"Expected shape (1, 10), got {output.shape}"
    print("Test passed successfully.")