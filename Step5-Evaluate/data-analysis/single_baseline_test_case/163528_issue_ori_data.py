# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch import nn

class Foo(nn.Module):
    def __init__(
        self,
        quantiles: torch.Tensor,
    ) -> None:
        super().__init__()
        assert quantiles.shape[0] > 0
        quantiles = quantiles.T
        self.q = nn.Parameter(quantiles, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(self.q, x.T).T
    

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

    print(f'device: {device}, diff: {torch.max(torch.abs(y_original - y_compiled)).item()}')
    print('orignal',y_original[:5, :5])
    print('compiled',y_compiled[:5, :5])
    

def main():
    batch_size = 32
    feature_dim = 10
    quantile_size = 100
    torch.manual_seed(42)
    x = torch.randn(batch_size, feature_dim, dtype=torch.float32)
    quantiles = torch.randn(quantile_size, feature_dim, dtype=torch.float32)
    quantiles = torch.sort(quantiles, dim=0)[0]
    test_device('cpu', x, quantiles)
    test_device('cuda', x, quantiles)

main()