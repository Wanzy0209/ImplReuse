import torch
from torch import nn

# Original Module from the bug report
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

# Helper function reflecting the logic of the similar API: 
# tf.compat.v1.metrics.mean_absolute_error
# Logic: mean(abs(y_true - y_pred), axis=-1)
def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred), dim=-1)

def test_searchsorted_compile():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return

    batch_size = 32
    feature_dim = 10
    quantile_size = 100
    torch.manual_seed(42)

    x = torch.randn(batch_size, feature_dim, dtype=torch.float32)
    quantiles = torch.randn(quantile_size, feature_dim, dtype=torch.float32)
    quantiles = torch.sort(quantiles, dim=0)[0]

    # Prepare data for GPU
    x_cuda = x.cuda()
    quantiles_cuda = quantiles.cuda()

    # Instantiate models
    foo_eager = Foo(quantiles_cuda).cuda()
    foo_compiled = torch.compile(Foo(quantiles_cuda).cuda(), fullgraph=True)

    # Warmup
    with torch.no_grad():
        _ = foo_eager(x_cuda)
        _ = foo_compiled(x_cuda)

    # Inference
    with torch.no_grad():
        y_eager = foo_eager(x_cuda)
        y_compiled = foo_compiled(x_cuda)

    # Calculate error using the pattern from the similar API (Mean Absolute Error)
    # We compare the eager output (ground truth) vs compiled output
    error = mean_absolute_error(y_eager, y_compiled)
    
    print(f"Max Mean Absolute Error: {torch.max(error).item()}")
    
    # Assert that the error is zero (or negligible)
    # The bug report indicates a mismatch, so this assertion would fail if the bug is present
    assert torch.all(error == 0), \
        f"torch.compile produced different results on GPU. Max MAE: {torch.max(error).item()}"

if __name__ == "__main__":
    test_searchsorted_compile()