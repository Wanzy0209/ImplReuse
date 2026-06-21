import torch
from torch import nn

class BucketizeModule(nn.Module):
    def __init__(
        self,
        boundaries: torch.Tensor,
    ) -> None:
        super().__init__()
        assert boundaries.shape[0] > 0
        # Mimicking the original logic where quantiles were transposed
        boundaries = boundaries.T
        self.b = nn.Parameter(boundaries, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replacing torch.searchsorted(self.q, x.T).T
        # with torch.bucketize(x.T, self.b).T
        # Note: bucketize(input, boundaries) vs searchsorted(sequence, input)
        return torch.bucketize(x.T, self.b).T
    

def test_device(device, x, boundaries):
    # Check if torch.compile is available (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print(f"torch.compile is not available (requires PyTorch 2.0+). Skipping test on {device}.")
        return

    x = x.to(device)
    boundaries = boundaries.to(device)
    
    # Eager model
    model_eager = BucketizeModule(boundaries).to(device)
    # Compiled model
    model_compiled = torch.compile(BucketizeModule(boundaries).to(device), fullgraph=True)

    # warm up
    with torch.no_grad():
        _ = model_eager(x)
        _ = model_compiled(x)

    # proper inference
    with torch.no_grad():
        y_eager = model_eager(x)
        y_compiled = model_compiled(x)

    diff = torch.max(torch.abs(y_eager - y_compiled)).item()
    print(f'device: {device}, diff: {diff}')
    
    # Assert that the compiled version produces the same result as the eager version
    assert torch.equal(y_eager, y_compiled), f"Results differ on {device}: {diff}"

def main():
    batch_size = 32
    feature_dim = 10
    boundary_size = 100
    torch.manual_seed(42)
    
    x = torch.randn(batch_size, feature_dim, dtype=torch.float32)
    boundaries = torch.randn(boundary_size, feature_dim, dtype=torch.float32)
    # Boundaries must be sorted for bucketize/searchsorted
    boundaries = torch.sort(boundaries, dim=0)[0]
    
    test_device('cpu', x, boundaries)
    
    if torch.cuda.is_available():
        test_device('cuda', x, boundaries)
    else:
        print("CUDA not available, skipping GPU test.")

if __name__ == "__main__":
    main()