import torch
import torch.nn as nn

class RegressionModel(nn.Module):
    """
    Minimal model reproducing the regression reported in Issue 160084.
    The model uses a state flag `first_batch` inside the forward pass.
    """
    def __init__(self, a=0, b=0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a).float())
        self.b = nn.Parameter(torch.tensor(b).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            # The logic involving state change is part of the trigger
            self.first_batch = False
        return x * self.a + self.b

def test_compile_inductor_with_state():
    """
    Tests torch.compile with the 'inductor' backend on a model method
    that modifies internal state during execution.
    
    This test targets the regression: RuntimeError: opt_ready_stream && opt_parent_stream
    which occurred on CUDA in PyTorch 2.8.
    """
    # The reported error is specific to CUDA streams
    if not torch.cuda.is_available():
        print("Skipping test: CUDA is not available.")
        return

    model = RegressionModel()
    
    # The specific regression pattern: compiling the bound method directly
    model.forward = torch.compile(model.forward, backend="inductor")

    inputs = torch.randn(4, 10).to("cuda")
    
    # Execute the model. In the bug report, this raised:
    # RuntimeError: opt_ready_stream && opt_parent_stream INTERNAL ASSERT FAILED
    try:
        output = model(inputs)
        
        # Basic assertion to ensure execution completed and returned a tensor
        assert output is not None
        assert output.shape == (4, 10)
        assert output.dtype == torch.float32
        
        print("Test Passed: torch.compile with inductor backend handled stateful forward pass correctly.")
        
    except RuntimeError as e:
        if "opt_ready_stream && opt_parent_stream" in str(e):
            print(f"Regression detected: {e}")
            raise
        else:
            # Re-raise if it's a different RuntimeError
            raise

if __name__ == "__main__":
    test_compile_inductor_with_state()