import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Reproduce the environment setup from the original issue
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage as per the bug report
device = "cpu"
print(f"Running on device: {device}")

# Generate dummy 3D data suitable for ConvTranspose3d
# MNIST is 2D, but ConvTranspose3d requires 5D input (N, C, D, H, W)
# We create synthetic data to mimic the batch loading process
batch_size = 8
input_channels = 3
depth, height, width = 16, 16, 16

# Create random tensors for input and target
dummy_input = torch.randn(batch_size, input_channels, depth, height, width)
# Target matches the output shape of the network (upscaled)
dummy_target = torch.randn(batch_size, input_channels, depth * 2, height * 2, width * 2)

train_data = TensorDataset(dummy_input, dummy_target)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

# Define a model using the Similar API: torch.nn.ConvTranspose3d
# This replaces the nn.Linear layers from the original MNIST example
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # ConvTranspose3d layer 1: Upsamples by stride 2
            nn.ConvTranspose3d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # ConvTranspose3d layer 2: Maintains dimensions
            nn.ConvTranspose3d(in_channels=16, out_channels=input_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.layers(x)

model = Net().to(device)
loss_fn = nn.MSELoss() # Using MSELoss for regression-like output
opt = torch.optim.SGD(model.parameters(), lr=0.01)

print("=== TRAIN ===")
# Execute one training step to populate gradients and optimizer states
# This mimics the state before the crash in the original issue
for X, y in train_loader:
    X, y = X.to(device), y.to(device)
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    break

print("=== EVAL (CRASH CHECK) ===")
# Execute evaluation step with no_grad
# The original bug reported a kernel restart here
try:
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_loader):
            print(f"Processing batch {batch}")
            X, y = X.to(device), y.to(device)
            pred = model(X)  # Potential crash point for the underlying bug
            break
    print("Test passed: No crash during evaluation.")
except RuntimeError as e:
    print(f"RuntimeError encountered: {e}")
except Exception as e:
    print(f"Unexpected exception: {e}")