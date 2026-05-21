import torch
import torch.nn as nn
import torch.export

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

# Setup device (using cuda to mimic the original context of a specific backend)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleModel().to(device)

batch_size = 20
input_data = torch.randn(batch_size, 10).to(device)

# Adapted call site: using torch.export.export instead of DataParallel
# We wrap the call in a try-except to mimic the original structure's error handling logic
try:
    # torch.export.export captures the computation graph
    exported_program = torch.export.export(model, args=(input_data,))
    
    # Verify the exported program works
    output = exported_program(input_data)
    
    # Basic assertion to ensure functionality
    assert output.shape == (batch_size, 5), f"Expected shape {(batch_size, 5)}, got {output.shape}"
    print("success")
except Exception as e:
    print(f"Test failed with error: {e}")
    raise