import torch
import time
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Adapted call: torch.diagflat instead of torch.unique
        # torch.diagflat creates a 2D tensor with the flattened input on the diagonal.
        # Unlike unique, the output shape depends on the input shape, not the data values.
        _x = torch.diagflat(x)
        
        # torch.diagflat returns a single tensor, so we adjust the return statement
        # to match the expected output type of the model logic.
        return self.relu(_x)

def GetInput():
    return torch.randn(8)

def run_once(model, x, label):
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    print(f"[{label}] ok, type={type(y)} time={(time.perf_counter()-t0)*1000:.3f}ms")
    return y

def main():
    print("torch.__version__ =", torch.__version__)

    model = MyModel()
    x = GetInput()

    # 1. Run Eager Mode
    print("Running Eager Mode...")
    y_eager = run_once(model, x, "Eager")

    # 2. Run Compiled Mode (fullgraph=True)
    # This verifies if torch.diagflat triggers the same "Dynamic shape operator" error
    # as torch.unique when fullgraph is enabled.
    print("Running Compiled Mode (fullgraph=True)...")
    try:
        compiled_model = torch.compile(model, fullgraph=True)
        y_compiled = run_once(compiled_model, x, "Compiled")

        # Verify that the compiled output matches the eager output
        assert torch.allclose(y_eager, y_compiled), "Outputs differ between eager and compiled modes"
        print("Test Passed: torch.diagflat works in fullgraph mode.")

    except torch._dynamo.exc.Unsupported as e:
        print(f"Test Failed: torch.diagflat rejected in fullgraph mode.")
        print(f"Error: {e}")
    except Exception as e:
        print(f"Test Failed with unexpected error: {e}")

if __name__ == "__main__":
    main()