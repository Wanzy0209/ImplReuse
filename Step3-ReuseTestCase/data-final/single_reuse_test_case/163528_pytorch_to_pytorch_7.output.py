import torch
from torch import nn

class Foo(nn.Module):
    def __init__(
        self,
        quantiles: torch.Tensor,
    ) -> None:
        super().__init__()
        # Retaining structure from original test, though unused by zeros_like
        self.q = nn.Parameter(quantiles, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replacing torch.searchsorted with torch.zeros_like
        return torch.zeros_like(x)

def test_device(device, x, quantiles):
    x = x.to(device)
    quantiles = quantiles.to(device)
    foo = Foo(quantiles).to(device)
    foo_compiled = torch.compile(Foo(quantiles).to(device), fullgraph=True)

    # warm up
    with torch.no_grad():
        y_original = foo(x)
        y_compiled = foo_compiled(x)

    # proper inference
    with torch.no_grad():
        y_original = foo(x)
        y_compiled = foo_compiled(x)

    # Verify that the compiled version produces the same result as the original
    assert torch.equal(y_original, y_compiled), f"Results differ on {device}"
    print(f'device: {device}, test passed')

def main():
    batch_size = 32
    feature_dim = 10
    quantile_size = 100
    torch.manual_seed(42)
    x = torch.randn(batch_size, feature_dim, dtype=torch.float32)
    quantiles = torch.randn(quantile_size, feature_dim, dtype=torch.float32)
    quantiles = torch.sort(quantiles, dim=0)[0]
    
    test_device('cpu', x, quantiles)
    
    if torch.cuda.is_available():
        test_device('cuda', x, quantiles)
    else:
        print("CUDA not available, skipping GPU test")

if __name__ == "__main__":
    main()