import torch
import pytest

class RegressionModel(torch.nn.Module):
    def __init__(self, a=0, b=0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a).float())
        self.b = torch.nn.Parameter(torch.tensor(b).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            # Leverage the similar API to check environment state within the model execution
            # This helps verify if distributed availability interacts with the stream issue
            dist_available = torch.distributed.is_available()
            print(f"Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}. Distributed available: {dist_available}")
            self.first_batch = False
        return x * self.a + self.b

def test_compile_inductor_regression():
    """
    Test case for Issue 160084: Regression on compile with backend inductor.
    Preserves the original bug reproduction logic while leveraging 
    torch.distributed.is_available to check environment state.
    """
    # Check CUDA availability as the bug is specific to CUDA streams
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping test")
        
    # Leverage the similar API: torch.distributed.is_available
    # We check this to ensure the test environment is aware of distributed state,
    # which might interact with CUDA streams and the inductor backend.
    is_dist_avail = torch.distributed.is_available()
    print(f"Running test with torch.distributed.is_available() = {is_dist_avail}")

    model = RegressionModel()
    torch_device = "cuda"
    
    # The core of the regression: compiling forward with inductor backend
    model.forward = torch.compile(model.forward, backend="inductor")
    
    inputs = torch.randn(4, 10).to(torch_device)
    
    # This call triggers the RuntimeError in the buggy version:
    # RuntimeError: opt_ready_stream && opt_parent_stream INTERNAL ASSERT FAILED
    _ = model(inputs)