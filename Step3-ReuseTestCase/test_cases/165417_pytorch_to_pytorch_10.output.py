import torch
import torch.nn as nn
import time

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Adaptation: Use torch.triu_indices instead of torch.unique
        # To simulate the dynamic shape scenario where output depends on input,
        # we derive the matrix dimensions (n, m) from the input tensor x.
        # Here we assume x contains the dimensions as data.
        n = int(x[0].item())
        m = int(x[1].item())

        # torch.triu_indices returns a tuple of tensors (row_indices, col_indices)
        _r, _c = torch.triu_indices(n, m)
        
        # Keep it simple like the original repro
        _r = _r.clone().detach()
        _c = _c.clone().detach()
        
        # Return tuple output
        return self.relu(_r), _c

def GetInput():
    # Input tensor containing dimensions for the upper triangle (e.g., 5x5)
    return torch.tensor([5.0, 5.0])

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

    # Run in eager mode
    run_once(model, x, "Eager")

    # Run in compiled mode (fullgraph)
    # This checks if torch.triu_indices is rejected as a dynamic shape operator
    try:
        compiled_model = torch.compile(model, fullgraph=True)
        run_once(compiled_model, x, "Compiled")
    except Exception as e:
        print(f"[Compiled] Error: {e}")

if __name__ == "__main__":
    main()