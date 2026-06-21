import torch
import torch.nn as nn

class TestModel(nn.Module):
    def forward(self, x):
        # Convert to sparse CSR format because 'aten::all' is not implemented for SparseCPU (COO) backend
        x_sparse = x.to_sparse_csr()
        # Use torch.all
        result = torch.all(x_sparse)
        return result

# Create a random tensor
x = torch.randn(10, 10)

model = TestModel()

# Test eager mode
eager_output = model(x)
print("Eager output:", eager_output)

# Test compiled mode
# Note: The original bug mentions backend='inductor' is default for torch.compile
try:
    compiled_output = torch.compile(model)(x)
    print("Compiled output:", compiled_output)

    # Verify outputs match
    assert eager_output == compiled_output, "Outputs do not match!"
except Exception as e:
    print(f"Error during compilation/execution: {e}")