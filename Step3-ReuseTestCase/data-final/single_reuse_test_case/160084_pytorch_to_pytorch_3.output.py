import torch
from torch import Tensor
import torch.library

# Define a custom operator to test the abstract implementation
def custom_linear_meta(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    # Abstract implementation: defines shape/dtype without data
    return torch.empty_like(x)

def custom_linear_impl(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    # Concrete implementation
    return x * a + b

# Define the library and operator
torch.library.define("testlib::custom_linear", "(Tensor x, Tensor a, Tensor b) -> Tensor")

# Register the abstract implementation using the API under test
torch.library.impl_abstract("testlib::custom_linear", custom_linear_meta)

# Register concrete implementations for CPU and CUDA
torch.library.impl("testlib::custom_linear", custom_linear_impl, "CPU")
if torch.cuda.is_available():
    torch.library.impl("testlib::custom_linear", custom_linear_impl, "CUDA")

class RegressionModel(torch.nn.Module):
    def __init__(self, a=0, b=0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a).float())
        self.b = torch.nn.Parameter(torch.tensor(b).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            # print(f"Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}")
            self.first_batch = False
        # Use the custom operator instead of standard ops to test the abstract impl
        return torch.ops.testlib.custom_linear(x, self.a, self.b)

def test_compile_with_custom_abstract_impl():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA specific test.")
        return

    torch_device = "cuda"
    model = RegressionModel().to(torch_device)

    # Compile the model using the inductor backend
    # This tests if the abstract implementation integrates correctly with torch.compile
    model.forward = torch.compile(model.forward, backend="inductor")

    inputs = torch.randn(4, 10).to(torch_device)
    
    # Run the model
    try:
        output = model(inputs)
        assert output is not None
        assert output.shape == (4, 10)
        print("Test passed: torch.compile works with torch.library.impl_abstract")
    except RuntimeError as e:
        if "opt_ready_stream && opt_parent_stream" in str(e):
            print(f"Regression detected: {e}")
            raise
        raise

if __name__ == "__main__":
    test_compile_with_custom_abstract_impl()