import torch
import torch.nn as nn
import sys

class SearchSortedModule(nn.Module):
    """
    Module wrapper for torch.searchsorted to test compilation behavior.
    This mirrors the structure of the reported issue.
    """
    def __init__(self, quantiles: torch.Tensor) -> None:
        super().__init__()
        assert quantiles.shape[0] > 0, "Quantiles must not be empty"
        # Transpose quantiles to match the issue's logic
        self.q = nn.Parameter(quantiles.T, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform searchsorted with transposes as in the original bug report
        return torch.searchsorted(self.q, x.T).T

def test_searchsorted_compiled():
    """
    Test case to verify torch.searchsorted behavior with torch.compile.
    Checks consistency between eager execution and compiled execution on CPU and CUDA.
    """
    # Check if torch.compile is available (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("Skipping test: torch.compile is not available (requires PyTorch 2.0+).")
        return

    batch_size = 32
    feature_dim = 10
    quantile_size = 100
    
    # Setup reproducible random data
    torch.manual_seed(42)
    x = torch.randn(batch_size, feature_dim, dtype=torch.float32)
    quantiles = torch.randn(quantile_size, feature_dim, dtype=torch.float32)
    # Sort quantiles as required by searchsorted
    quantiles = torch.sort(quantiles, dim=0)[0]

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        print(f"Testing on device: {device}")
        x_dev = x.to(device)
        quantiles_dev = quantiles.to(device)

        # Initialize standard and compiled models
        model = SearchSortedModule(quantiles_dev).to(device)
        # Using fullgraph=True as in the original bug report
        compiled_model = torch.compile(SearchSortedModule(quantiles_dev).to(device), fullgraph=True)

        # Warmup runs to allow compilation to occur
        with torch.no_grad():
            _ = model(x_dev)
            _ = compiled_model(x_dev)

        # Actual inference run
        with torch.no_grad():
            y_original = model(x_dev)
            y_compiled = compiled_model(x_dev)

        # Calculate difference
        diff = torch.max(torch.abs(y_original - y_compiled)).item()
        
        print(f"  Max difference between eager and compiled: {diff}")
        
        # Assert that the results are identical
        # Note: If the bug is present, this assertion will fail on CUDA.
        assert diff == 0.0, (
            f"torch.searchsorted produced different results with torch.compile on {device}. "
            f"Max diff: {diff}"
        )
        
        # Optional: Print sample outputs for visual verification
        if diff > 0:
            print("  Original (first 5, first 5):", y_original[:5, :5])
            print("  Compiled (first 5, first 5):", y_compiled[:5, :5])

if __name__ == "__main__":
    try:
        test_searchsorted_compiled()
        print("\nTest passed successfully.")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)