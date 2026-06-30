import torch
import collections

# Leveraging the similar API pattern (tf.compat.v1.train.SessionRunValues)
# to structure the execution results of the PyTorch model.
# SessionRunValues typically contains: results, options, run_metadata.
class TorchRunValues(collections.namedtuple("TorchRunValues", 
                                            ["results", "options", "run_metadata"])):
    """Contains the results of a compiled model run, mirroring tf.compat.v1.train.SessionRunValues."""
    pass

class RegressionModel(torch.nn.Module):
    """
    Original model from the bug report.
    The bug involves a regression in torch.compile (inductor backend) 
    when handling stateful logic (self.first_batch) mixed with tensor operations.
    """
    def __init__(self, a=0, b=0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a).float())
        self.b = torch.nn.Parameter(torch.tensor(b).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            # This side-effect and control flow triggered the INTERNAL ASSERT FAILED
            # in PyTorch 2.8 with backend="inductor".
            print(f"Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}")
            self.first_batch = False
        return x * self.a + self.b

def test_compile_inductor_with_session_like_values():
    """
    Test case for Issue 160084.
    Reproduces the regression on torch.compile with backend inductor.
    Uses the SessionRunValues pattern to encapsulate the run results.
    """
    # Fix: Check if torch.compile exists (PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("torch.compile is not available (requires PyTorch 2.0+), skipping test.")
        return

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Setup model and device
    model = RegressionModel()
    torch_device = "cuda"
    model.to(torch_device)

    # Apply torch.compile with the problematic backend
    # This is the core of the reported regression.
    model.forward = torch.compile(model.forward, backend="inductor")

    # Prepare inputs
    inputs = torch.randn(4, 10).to(torch_device)

    # Execute the compiled model
    # In torch 2.8, this line raised:
    # RuntimeError: opt_ready_stream && opt_parent_stream INTERNAL ASSERT FAILED
    output = model(inputs)

    # Leverage the similar API (SessionRunValues) to structure the verification.
    # We map the PyTorch execution context to the TensorFlow SessionRunValues fields.
    run_values = TorchRunValues(
        results=output,
        options={"backend": "inductor", "mode": "torch.compile"},
        run_metadata={"device": torch_device, "state_changed": not model.first_batch}
    )

    # Assertions to verify correctness and successful execution
    assert run_values.results is not None, "Model output should not be None"
    assert run_values.results.shape == (4, 10), f"Expected shape (4, 10), got {run_values.results.shape}"
    assert run_values.run_metadata["state_changed"], "Model state (first_batch) should have changed"
    
    print("Test passed successfully.")

if __name__ == "__main__":
    test_compile_inductor_with_session_like_values()