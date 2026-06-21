import torch

class MaskedSelectScalar(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(777)
        self.device = device
        # Input tensor
        self.x = torch.randint(-50, 50, (10,), dtype=torch.int64, device=device)
        # Scalar mask (0-d tensor)
        # Using a boolean scalar to mimic the scalar input pattern of the original bug
        self.y = torch.tensor(True, dtype=torch.bool, device=device)

    def forward(self):
        print(self.x)
        print(self.y)
        # torch.masked_select with a scalar mask
        out = torch.masked_select(self.x, self.y)
        return {'out': out}

# Check for CUDA availability to ensure the test is runnable
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")

model = MaskedSelectScalar(device)
print("Eager:", model.forward())

# Check if torch.compile is available (requires PyTorch 2.0+)
if hasattr(torch, 'compile'):
    # Compile with inductor
    compiled_model = torch.compile(model, backend='inductor')
    print("Inductor:", compiled_model.forward())

    # Verify results match
    eager_out = model.forward()['out']
    compiled_out = compiled_model.forward()['out']
    assert torch.equal(eager_out, compiled_out), "Eager and Inductor outputs do not match"
    print("Test passed!")
else:
    print("torch.compile is not available (requires PyTorch 2.0+). Skipping compilation test.")