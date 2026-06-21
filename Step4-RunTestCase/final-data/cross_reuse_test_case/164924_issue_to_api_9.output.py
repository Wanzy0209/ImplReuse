import torch
from torch.library import Library, impl

# Define a custom library and operation to mimic the scenario where 
# torch.library.register_kernel is used to handle specific device logic.
# This pattern is similar to how internal fixes for scalar inputs are often implemented.
lib = Library("test_isin_scalar", "DEF")
lib.define("custom_isin(Tensor elements, Tensor test_elements) -> Tensor")

# Register kernel for CPU
@impl(lib, "custom_isin", "CPU")
def custom_isin_cpu(elements, test_elements):
    # Simplified logic: check if elements are in test_elements
    # For the scalar case (0-dim test_elements), this behaves like equality check
    return elements == test_elements

# Register kernel for CUDA
@impl(lib, "custom_isin", "CUDA")
def custom_isin_cuda(elements, test_elements):
    return elements == test_elements

class TestIsinScalar(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(777)
        self.device = device
        # Reproduce the exact setup from the original bug report
        self.x = torch.randint(-50, 50, (1,), dtype=torch.int64, device=device)
        self.y = torch.randint(-50, 50, (), dtype=torch.int64, device=device)

    def forward(self):
        # Use the custom op that relies on the registered kernel
        out = torch.ops.test_isin_scalar.custom_isin(self.x, self.y)
        return out

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TestIsinScalar(device)
    
    # Eager execution
    print("Eager:", model.forward())
    
    # Check for torch.compile availability (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        # Compiled execution
        compiled_model = torch.compile(model, backend='inductor')
        print("Inductor:", compiled_model.forward())
        
        # Verify consistency
        eager_out = model.forward()
        compiled_out = compiled_model.forward()
        assert torch.equal(eager_out, compiled_out), "Eager and Compiled outputs must match"
        print("Test passed.")
    else:
        print("Skipping compiled execution: torch.compile is not available (requires PyTorch 2.0+).")