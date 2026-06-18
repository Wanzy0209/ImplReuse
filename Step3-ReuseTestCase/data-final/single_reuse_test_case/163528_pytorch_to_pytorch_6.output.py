import torch
from torch import nn

class Foo(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replacing torch.searchsorted with torch.nn.functional.dropout
        return torch.nn.functional.dropout(x, p=self.p, training=True)

def test_device(device, x):
    x = x.to(device)
    foo = Foo(p=0.5).to(device)
    foo_compiled = torch.compile(Foo(p=0.5).to(device), fullgraph=True)

    # warm up
    with torch.no_grad():
        _ = foo(x)
        _ = foo_compiled(x)

    # proper inference
    # Reset seed to ensure reproducibility for comparison
    torch.manual_seed(42)
    with torch.no_grad():
        y_original = foo(x)

    torch.manual_seed(42)
    with torch.no_grad():
        y_compiled = foo_compiled(x)

    diff = torch.max(torch.abs(y_original - y_compiled)).item()
    print(f'device: {device}, diff: {diff}')
    print('original', y_original[:5, :5])
    print('compiled', y_compiled[:5, :5])
    
    # Assert that the outputs match. 
    # For dropout, with the same seed, the random mask should be identical 
    # in both eager and compiled modes.
    assert diff == 0.0, f"Outputs differ on {device}"

def main():
    batch_size = 32
    feature_dim = 10
    torch.manual_seed(42)
    x = torch.randn(batch_size, feature_dim, dtype=torch.float32)

    if torch.cuda.is_available():
        test_device('cuda', x)
    test_device('cpu', x)

if __name__ == "__main__":
    main()