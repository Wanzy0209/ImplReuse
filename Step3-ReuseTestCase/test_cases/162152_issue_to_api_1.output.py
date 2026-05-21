import torch
import torch.nn as nn
from torch.nn import DataParallel

# Define a model that leverages the similar API (torch.nn.Sigmoid)
# and follows the structural pattern of the provided Sigmoid implementation.
class SigmoidModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SigmoidModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Reusing the similar API: torch.nn.Sigmoid
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x

def test_dataparallel_sigmoid():
    # Preserving the original bug reproduction logic:
    # Check for device availability (using CUDA as a runnable proxy for <mybackend>)
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    device_count = torch.cuda.device_count()
    if device_count < 2:
        print(f"Only {device_count} GPU(s) available, need at least 2 for DataParallel. Skipping test.")
        return

    print(f"Detected {device_count} GPUs")

    model = SigmoidModel()
    model = model.cuda()  # Move to GPU 0

    # Use DataParallel as in the original bug report
    # Using available devices to ensure runnability
    device_ids = list(range(device_count))
    model = DataParallel(model, device_ids=device_ids)

    batch_size = 20
    input_data = torch.randn(batch_size, 10).cuda()

    # Run the model
    output = model(input_data)
    
    # Assertions to verify the test logic
    assert output is not None, "Output is None"
    assert output.shape == (batch_size, 5), f"Expected output shape ({batch_size}, 5), got {output.shape}"
    
    print("Test passed: DataParallel with Sigmoid model executed successfully.")

if __name__ == "__main__":
    test_dataparallel_sigmoid()