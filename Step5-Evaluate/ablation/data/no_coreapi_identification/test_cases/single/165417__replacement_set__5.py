import torch
import torch.nn.functional as F
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        # The API under test: torch.unique
        # _unique2.default under the hood
        _x, _i = torch.unique(x, sorted=True, return_inverse=True)
        _x = _x.clone().detach()   # keep it simple
        return self.relu(_x), _i   # tuple output

def GetInput():
    # Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool3d
    # We create a dummy input tensor to apply pooling to, simulating input generation.
    # Using a constant tensor to ensure determinism since we cannot use torch.randn.
    dummy_input = torch.ones(1, 1, 4, 4, 4) 
    # Apply adaptive max pool to get a specific output size (2, 2, 2) -> 8 elements
    # This replaces the random generation of a tensor of size 8.
    pooled = F.adaptive_max_pool3d(dummy_input, output_size=(2, 2, 2))
    return pooled.flatten()

def run_once(model, x, label):
    model.eval()
    with torch.no_grad():
        y = model(x)
    
    # Replacement: torch.compile -> torch.all
    # Instead of compiling the model, we verify the output using torch.all.
    # We check if all elements in the first output tensor (ReLU result) are non-negative.
    is_valid = torch.all(y[0] >= 0)
    
    print(f"[{label}] ok, types={[type(t) for t in (y if isinstance(y, (tuple, list)) else [y])]} "
          f"check_passed={is_valid.item()}")
    return y

def main():
    print("torch.__version__ =", torch.__version__)

    model = MyModel()
    x = GetInput()
    
    # Run the model in eager mode
    run_once(model, x, "eager")

if __name__ == "__main__":
    main()