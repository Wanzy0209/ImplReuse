import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Adapted to use torch.fliplr instead of torch.unique
        # torch.fliplr flips the tensor in the left/right direction.
        # It requires at least 2-dimensional input.
        _x = torch.fliplr(x)
        return self.relu(_x)

def GetInput():
    # torch.fliplr requires a 2D tensor (N, M)
    return torch.randn(4, 8)

def run_once(model, x, label):
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"[{label}] ok, output shape={y.shape} type={type(y)}")
    return y

def main():
    print("torch.__version__ =", torch.__version__)

    model = MyModel()
    x = GetInput()

    # 1. Run Eager
    y_eager = run_once(model, x, "Eager")

    # 2. Run torch.compile with fullgraph=True
    # The original bug report indicated that torch.unique failed here with 
    # "Dynamic shape operator". We verify if torch.fliplr works or fails similarly.
    print("\nAttempting torch.compile(fullgraph=True)...")
    try:
        compiled_model = torch.compile(model, fullgraph=True)
        y_compiled = run_once(compiled_model, x, "Compiled")
        
        # Verify correctness
        if torch.allclose(y_eager, y_compiled):
            print("Test Passed: torch.fliplr works correctly in fullgraph mode.")
        else:
            print("Test Failed: Output mismatch.")
            
    except Exception as e:
        print(f"Test Failed with Exception: {e}")

if __name__ == "__main__":
    main()