import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import swa_utils

# Fix: Check availability instead of asserting, fallback to CPU if MPS is not available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Define a model with BatchNorm layers to test update_bn
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32)
        )

    def forward(self, x):
        return self.features(x)

model = TestModel().to(device)

# Create a dummy data loader
# Data needs to match the input shape of the model (Batch, Channel, Height, Width)
dummy_data = torch.randn(10, 3, 32, 32)
dataset = TensorDataset(dummy_data)
loader = DataLoader(dataset, batch_size=2)

# Call the similar API: torch.optim.swa_utils.update_bn
# This replaces the original usage of the device with the specific utility function
swa_utils.update_bn(loader, model, device=device)

print("Test completed successfully.")