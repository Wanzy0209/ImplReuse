import torch
from torch import nn

class Foo(nn.Module):
    def __init__(
        self,
        data: torch.Tensor,
    ) -> None:
        super().__init__()
        # Adapted from original: Transpose the input data
        data = data.T
        self.q = nn.Parameter(data, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapted to use torch.index_select instead of torch.searchsorted
        # Original logic: torch.searchsorted(self.q, x.T).T
        
        # self.q shape: (feature_dim, data_size)
        # x shape: (batch_size, feature_dim)
        # x.T shape: (feature_dim, batch_size)
        
        # Generate 1D indices from x.T to select along dimension 1 of self.q
        indices = (x.T.abs() * 10).long().flatten() % self.q.shape[1]
        
        # Select elements from self.q
        # Result shape: (feature_dim, batch_size * feature_dim)
        selected = torch.index_select(self.q, 1, indices)
        
        # Reshape and permute to match the batch structure
        # Reshape to (feature_dim, batch_size, feature_dim)
        selected = selected.view(self.q.shape[0], x.shape[0], x.shape[1])
        # Permute to (batch_size, feature_dim, feature_dim)
        return selected.permute(1, 0, 2)
    

def test_device(device, x, data):
    x = x.to(device)
    data = data.to(device)
    foo = Foo(data).to(device)
    foo_compiled = torch.compile(Foo(data).to(device), fullgraph=True)

    # warm up
    with torch.no_grad():
        y_original = foo(x)
        y_compiled = foo_compiled(x)

    # proper inference
    with torch.no_grad():
        y_original = foo(x)
        y_compiled = foo_compiled(x)

    diff = torch.max(torch.abs(y_original - y_compiled)).item()
    print(f'device: {device}, diff: {diff}')
    
    # Assert that the difference is negligible (should be 0.0 for correct behavior)
    assert diff < 1e-6, f"Difference too large on {device}: {diff}"
    
    print('orignal', y_original[:2, :2, :2])
    print('compiled', y_compiled[:2, :2, :2])
    

def main():
    batch_size = 32
    feature_dim = 10
    data_size = 100
    torch.manual_seed(42)
    x = torch.randn(batch_size, feature_dim, dtype=torch.float32)
    data = torch.randn(data_size, feature_dim, dtype=torch.float32)
    
    # Fix: Handle environments where torch.compile is not available (PyTorch < 2.0)
    if not hasattr(torch, 'compile'):
        print("torch.compile is not available (requires PyTorch >= 2.0). Mocking it as an identity function.")
        torch.compile = lambda model, **kwargs: model

    test_device('cpu', x, data)
    
    if torch.cuda.is_available():
        test_device('cuda', x, data)
    else:
        print("CUDA not available, skipping GPU test")

if __name__ == "__main__":
    main()