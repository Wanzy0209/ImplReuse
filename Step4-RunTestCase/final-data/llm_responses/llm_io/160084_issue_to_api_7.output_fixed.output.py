import torch
import sys

class RegressionModel(torch.nn.Module):
    def __init__(self, a=0, b=0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a).float())
        self.b = torch.nn.Parameter(torch.tensor(b).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            # Preserving the original print logic for debugging context
            print(f"Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}")
            self.first_batch = False
        return x * self.a + self.b

def test_torch_compile_inductor_regression():
    """
    Test case for Issue 160084: Regression on compile with backend inductor with torch 2.8.
    
    This test leverages the semantic pattern of the similar API 'tf.test.is_built_with_rocm'.
    Just as TensorFlow tests check for ROCm support before executing GPU-specific code,
    this test checks for CUDA availability using 'torch.cuda.is_available()' to ensure
    the environment supports the execution context required to reproduce the bug.
    """
    
    # Translating the semantic of tf.test.is_built_with_rocm to PyTorch:
    # Ensure the test only runs if the necessary GPU backend (CUDA) is available.
    if not torch.cuda.is_available():
        print("Test skipped: CUDA is not available (analogous to !tf.test.is_built_with_rocm).")
        return

    # Check if torch.compile is available (introduced in PyTorch 2.0)
    if not hasattr(torch, 'compile'):
        print("Test skipped: torch.compile is not available (requires PyTorch >= 2.0).")
        return

    try:
        model = RegressionModel()
        torch_device = "cuda"
        
        # The specific configuration causing the regression in torch 2.8
        model.forward = torch.compile(model.forward, backend="inductor")
        
        inputs = torch.randn(4, 10).to(torch_device)
        
        # Execute the forward pass
        output = model(inputs)
        
        # Assertions to verify correct execution
        assert output is not None, "Model output is None"
        assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
        assert output.device.type == torch_device, f"Output not on {torch_device}"
        
        print("Test passed: torch.compile with inductor backend on CUDA succeeded.")
        
    except RuntimeError as e:
        # Catching the specific internal assertion failed error mentioned in the bug report
        if "opt_ready_stream && opt_parent_stream" in str(e):
            print(f"Regression detected (Issue 160084): {e}")
            sys.exit(1)
        else:
            # Re-raise if it's a different RuntimeError
            raise

if __name__ == "__main__":
    test_torch_compile_inductor_regression()