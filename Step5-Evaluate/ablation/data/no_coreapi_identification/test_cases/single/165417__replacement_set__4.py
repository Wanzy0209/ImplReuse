import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.library import Library

# --- Replacements ---
# 1. torch.compile -> torch.library.impl_abstract
# We define a custom library and operator to utilize torch.library.impl_abstract.
# This replaces the compilation step with an abstract implementation registration step.
lib = Library("my_unique_lib", "DEF")
lib.define("my_unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)")

def my_unique_impl(x, sorted=True, return_inverse=False):
    # The implementation still uses torch.unique
    return torch.unique(x, sorted=sorted, return_inverse=return_inverse)

def my_unique_abstract(x, sorted=True, return_inverse=False):
    # Abstract implementation defining shape behavior
    # Since unique is dynamic, we return a shape matching the input for this test
    return x, x

lib.impl("my_unique", my_unique_impl, "CPU")
torch.library.impl_abstract("my_unique_lib::my_unique", my_unique_abstract)
# ---------------------

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Using the custom operator which wraps torch.unique
        _x, _i = torch.ops.my_unique_lib.my_unique(x, sorted=True, return_inverse=True)
        _x = _x.clone().detach()
        return self.relu(_x), _i

def GetInput():
    # 2. torch.randn -> torch.nn.functional.adaptive_max_pool2d
    # We generate the input tensor using adaptive_max_pool2d instead of randn.
    # Original: torch.randn(8) -> Shape (8,)
    # Replacement: Create a dummy 4D tensor, pool it to (1, 8), and flatten to (8,)
    dummy_input = torch.randn(1, 1, 8, 8)
    pooled = F.adaptive_max_pool2d(dummy_input, (1, 8))
    return pooled.flatten()

def run_once(model, x, label):
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"[{label}] ok, types={[type(t) for t in (y if isinstance(y, (tuple, list)) else [y])]}")
    return y

def main():
    print("torch.__version__ =", torch.__version__)
    
    model = MyModel()
    x = GetInput()
    
    # Run eager mode
    run_once(model, x, "eager")
    
    # Note: The original torch.compile call was replaced by torch.library.impl_abstract
    # registration above. We verify the model runs with the custom operator.

if __name__ == "__main__":
    main()