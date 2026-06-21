import torch
from torch import nn

# Compatibility fix for PyTorch versions < 2.0
if not hasattr(torch, 'compile'):
    print("torch.compile is not available (requires PyTorch 2.0+). Using mock identity function.")
    # Define a mock to allow the script to run without error
    # It simply returns the model, so the test compares the model against itself
    torch.compile = lambda model, **kwargs: model

class FooMedian(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch.median returns a named tuple (values, indices) when dim is specified.
        # We extract the values to compare the tensor outputs.
        return torch.median(x, dim=1, keepdim=True).values

def test_device(device, x):
    x = x.to(device)
    foo = FooMedian().to(device)
    foo_compiled = torch.compile(FooMedian().to(device), fullgraph=True)

    # warm up
    with torch.no_grad():
        y_original = foo(x)
        y_compiled = foo_compiled(x)

    # proper inference
    with torch.no_grad():
        y_original = foo(x)
        y_compiled = foo_compiled(x)

    print(f'device: {device}, diff: {torch.max(torch.abs(y_original - y_compiled)).item()}')
    print('original', y_original[:5, :5])
    print('compiled', y_compiled[:5, :5])
    
    # Assert that the compiled version produces the same result as the original
    assert torch.allclose(y_original, y_compiled), f"Results differ on {device}"

def main():
    batch_size = 32
    feature_dim = 10
    torch.manual_seed(42)
    x = torch.randn(batch_size, feature_dim, dtype=torch.float32)
    
    test_device('cpu', x)
    
    if torch.cuda.is_available():
        test_device('cuda', x)
    else:
        print("CUDA not available, skipping GPU test")

if __name__ == "__main__":
    main()