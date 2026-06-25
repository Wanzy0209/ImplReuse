import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x):
        # Simulate dynamic tensor sizes that cause recompilation
        seq_len = x.size(1)
        if seq_len % 2 == 0:
            x = x[:, :seq_len-1]  # Change size to trigger recompile
        return self.linear(x)

model = SimpleModel()
compiled_model = torch.compile(model)

# Test with varying sequence lengths to trigger the bug
for i in range(10):
    x = torch.randn(1, 77 + i, 10)  # Varying sizes
    output = compiled_model(x)
    print(f"Iteration {i}, input shape: {x.shape}")