import torch
import time
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Adapted to use torch.flipud instead of torch.unique
        # torch.flipud flips the tensor in the up/down direction.
        # Unlike torch.unique, torch.flipud preserves the input shape,
        # so it should not trigger the "Dynamic shape operator" error.
        _x = torch.flipud(x)
        _x = _x.clone().detach()
        # torch.flipud returns a single tensor, so we return just the tensor result
        return self.relu(_x)

def GetInput():
    # torch.flipud works on 1D tensors (reverses the order)
    return torch.randn(8)

def run_once(model, x, label):
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    print(f"[{label}] ok, types={[type(t) for t in (y if isinstance(y, (tuple, list)) else [y])]} "
          f"time={(time.perf_counter()-t0)*1000:.3f}ms")
    return y

def main():
    print("torch.__version__ =", torch.__version__)

    model = MyModel()
    x = GetInput()

    # 1. Run Eager
    print("Running Eager...")
    y_eager = run_once(model, x, "Eager")

    # 2. Run torch.compile with fullgraph=True
    # The original bug report indicated torch.unique failed here with "Dynamic shape operator".
    # We verify if torch.flipud (which preserves shape) succeeds.
    print("Running torch.compile(fullgraph=True)...")
    try:
        compiled_model = torch.compile(model, fullgraph=True)
        y_compiled = run_once(compiled_model, x, "Compiled")
        
        # Verify results match
        assert torch.allclose(y_eager, y_compiled), "Outputs differ between eager and compiled!"
        print("Test Passed: torch.flipud works correctly with fullgraph=True")
    except Exception as e:
        print(f"Test Failed with error: {e}")

if __name__ == "__main__":
    main()