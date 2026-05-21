import torch
import torch.nn as nn
from torch.nn import DataParallel

class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def test_dataparallel_functionality():
    """
    Test case adapted from the bug report to verify torch.nn.DataParallel behavior.
    This test uses the standard CUDA backend to ensure the API logic functions correctly
    with multiple devices, mirroring the user's intended usage with a custom backend.
    """
    # Check for CUDA availability (standard backend for DataParallel)
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Ensure at least 2 devices are available for parallelism
    device_count = torch.cuda.device_count()
    if device_count < 2:
        print(f"Detected only {device_count} GPU(s). At least 2 required for DataParallel, skipping test.")
        return

    print(f"Detected {device_count} GPUs")

    # Initialize model and move to GPU 0
    model = SimpleModel()
    model = model.cuda()

    # Apply DataParallel using the first two available devices
    model = DataParallel(model, device_ids=[0, 1])

    # Prepare input data on GPU 0
    batch_size = 20
    input_data = torch.randn(batch_size, 10).cuda()

    # Forward pass
    output = model(input_data)

    # Assertions to verify the output
    assert output is not None, "Model output is None"
    assert output.device.type == 'cuda', "Output is not on CUDA device"
    # DataParallel gathers results on the primary device (device_ids[0])
    assert output.device.index == 0, "Output is not on the primary device (0)"
    assert output.shape == (batch_size, 5), f"Output shape mismatch: expected {(batch_size, 5)}, got {output.shape}"

    print("Test passed: DataParallel executed successfully.")

if __name__ == "__main__":
    test_dataparallel_functionality()