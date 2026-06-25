import torch
import torch._inductor

# Test with a simple model to reproduce regression
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 512)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel().cpu()
x = torch.randn(2, 512).cpu()

# Compile with inductor
compiled_model = torch.compile(model, backend='inductor')
result = compiled_model(x)
print('Compiled successfully')