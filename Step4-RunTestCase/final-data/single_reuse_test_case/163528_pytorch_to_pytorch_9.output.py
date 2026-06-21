import torch
from torch import nn

class BatchNormModel(nn.Module):
    def __init__(
        self,
        num_features: int,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        # Register buffers for running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        # Parameters for affine transformation
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Using inference mode (training=False) to match the stateless nature of the original test
        return torch.nn.functional.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False
        )
    

def test_device(device, x, num_features):
    # Check if torch.compile is available (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print(f"Skipping test on {device}: torch.compile is not available (requires PyTorch 2.0+)")
        return

    x = x.to(device)
    model = BatchNormModel(num_features).to(device)
    model_compiled = torch.compile(BatchNormModel(num_features).to(device), fullgraph=True)

    # warm up
    with torch.no_grad():
        y_original = model(x)
        y_compiled = model_compiled(x)

    # proper inference
    with torch.no_grad():
        y_original = model(x)
        y_compiled = model_compiled(x)

    diff = torch.max(torch.abs(y_original - y_compiled)).item()
    print(f'device: {device}, diff: {diff}')
    print('original', y_original[:5, :5])
    print('compiled', y_compiled[:5, :5])
    
    # Assert to ensure the compiled version produces the same result as the original
    assert torch.allclose(y_original, y_compiled, atol=1e-5), f"Results differ on {device}"

def main():
    batch_size = 32
    num_features = 10
    torch.manual_seed(42)
    x = torch.randn(batch_size, num_features, dtype=torch.float32)
    
    test_device('cpu', x, num_features)
    
    if torch.cuda.is_available():
        test_device('cuda', x, num_features)
    else:
        print("CUDA not available, skipping GPU test")

if __name__ == "__main__":
    main()