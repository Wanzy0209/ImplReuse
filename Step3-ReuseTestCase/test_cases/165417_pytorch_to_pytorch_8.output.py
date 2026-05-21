import torch
import torch.nn as nn
import time

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Adapted call: torch.unique -> torch.absolute
        # torch.absolute is a point-wise operation and preserves the input shape.
        # Unlike torch.unique, it does not produce data-dependent dynamic shapes.
        _x = torch.absolute(x)
        _x = _x.clone().detach()
        return self.relu(_x)

def GetInput():
    return torch.randn(8)

def run_once(model, x, label):
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    print(f"[{label}] ok, output type={type(y)}, shape={y.shape} "
          f"time={(time.perf_counter()-t0)*1000:.3f}ms")
    return y

def main():
    print("torch.__version__ =", torch.__version__)
    
    model = MyModel()
    x = GetInput()

    # 1. Run Eager
    print("--- Running Eager ---")
    y_eager = run_once(model, x, "Eager")

    # 2. Run torch.compile with fullgraph=True
    # The original bug report indicated torch.unique failed here with:
    # "torch._dynamo.exc.Unsupported: Dynamic shape operator"
    # We expect torch.absolute to succeed because it maintains static shapes.
    print("\n--- Running torch.compile(fullgraph=True) ---")
    try:
        compiled_model = torch.compile(model, fullgraph=True)
        y_compiled = run_once(compiled_model, x, "Compiled")
        
        # Verify correctness
        assert torch.allclose(y_eager, y_compiled), "Output mismatch between eager and compiled"
        assert y_eager.shape == x.shape, "Output shape must match input shape for static ops"
        
        print("\nTest Passed: torch.absolute works correctly in fullgraph mode.")
        
    except torch._dynamo.exc.Unsupported as e:
        print(f"\nTest Failed: {e}")
        raise

if __name__ == "__main__":
    main()