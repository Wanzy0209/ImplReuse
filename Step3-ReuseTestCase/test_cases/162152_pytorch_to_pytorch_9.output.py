import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, num_channels=10):
        super(SimpleModel, self).__init__()
        # Adapted to use Dropout2d which expects 4D input (N, C, H, W)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        return self.dropout(x)

# Test setup
# Dropout2d requires 4D input (Batch, Channels, Height, Width)
batch_size = 20
num_channels = 10
height, width = 5, 5
input_data = torch.randn(batch_size, num_channels, height, width)

model = SimpleModel()
# Set to train mode to ensure dropout is active
model.train() 

output = model(input_data)

# Verify output shape matches input shape (Dropout2d does not change shape)
assert output.shape == input_data.shape, f"Shape mismatch: {output.shape} != {input_data.shape}"
print("success")