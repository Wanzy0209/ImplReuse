import torch
import torch.nn as nn

def test_searchsorted_compile():
    """
    Regression test for Issue #163528: torch.compile don't work with searchsorted.
    Verifies that torch.searchsorted produces consistent results between eager and 
    compiled modes on both CPU and CUDA.
    """
    
    class SearchSortedModel(nn.Module):
        def __init__(self, quantiles: torch.Tensor) -> None:
            super().__init__()
            # Register as buffer to avoid moving it accidentally, or Parameter if requires_grad is needed
            # The bug report uses Parameter with requires_grad=False
            self.register_buffer('q', quantiles)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Replicating the logic from the bug report
            return torch.searchsorted(self.q, x.T).T

    # Setup data
    batch_size = 32
    feature_dim = 10
    quantile_size = 100
    torch.manual_seed(42)
    
    x = torch.randn(batch_size, feature_dim, dtype=torch.float32)
    quantiles = torch.randn(quantile_size, feature_dim, dtype=torch.float32)
    # searchsorted requires sorted sequence
    quantiles = torch.sort(quantiles, dim=0)[0]

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        print(f"Testing on {device}...")
        
        x_dev = x.to(device)
        q_dev = quantiles.to(device)

        model = SearchSortedModel(q_dev).to(device)
        # fullgraph=True is used in the bug report
        compiled_model = torch.compile(model, fullgraph=True)

        with torch.no_grad():
            # Warmup runs to ensure compilation happens before measurement
            _ = model(x_dev)
            _ = compiled_model(x_dev)

            # Actual inference
            y_original = model(x_dev)
            y_compiled = compiled_model(x_dev)

        # Verify results match
        diff = torch.max(torch.abs(y_original - y_compiled)).item()
        print(f"  Max difference: {diff}")
        
        # Assert that the difference is negligible (floating point precision)
        assert torch.allclose(y_original, y_compiled, atol=1e-5), \
            f"Failed on {device}: Results differ between eager and compiled execution."
        
        print(f"  Passed on {device}")

if __name__ == "__main__":
    test_searchsorted_compile()