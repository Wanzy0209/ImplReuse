import torch
import torch.nn as nn
import unittest

class SearchSortedModule(nn.Module):
    """
    Module that mimics the structure found in the bug report.
    The pattern of transposing inputs and outputs (axis manipulation)
    is structurally similar to the logic in tf.experimental.numpy.diagonal
    which uses moveaxis to align dimensions before processing.
    """
    def __init__(self, quantiles: torch.Tensor):
        super().__init__()
        # Transpose quantiles to match the specific axis manipulation pattern
        self.q = nn.Parameter(quantiles.T, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose input, perform searchsorted, then transpose output back.
        # This sequence of axis manipulations is the core of the reported issue.
        return torch.searchsorted(self.q, x.T).T

class TestTorchCompileSearchSorted(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.feature_dim = 10
        self.quantile_size = 100
        torch.manual_seed(42)
        
        # Generate sorted quantiles
        self.quantiles = torch.randn(self.quantile_size, self.feature_dim, dtype=torch.float32)
        self.quantiles = torch.sort(self.quantiles, dim=0)[0]
        
        # Generate input data
        self.x = torch.randn(self.batch_size, self.feature_dim, dtype=torch.float32)

    def _test_device(self, device_str):
        device = torch.device(device_str)
        x = self.x.to(device)
        quantiles = self.quantiles.to(device)
        
        # Initialize standard and compiled models
        model = SearchSortedModule(quantiles).to(device)
        model_compiled = torch.compile(SearchSortedModule(quantiles).to(device), fullgraph=True)

        with torch.no_grad():
            # Warmup runs
            _ = model(x)
            _ = model_compiled(x)
            
            # Actual inference runs
            y_original = model(x)
            y_compiled = model_compiled(x)

        # Calculate difference
        diff = torch.max(torch.abs(y_original - y_compiled)).item()
        
        # Assert that compiled and eager execution match
        # This assertion will fail on CUDA if the bug (Issue 163528) is present
        self.assertTrue(
            torch.allclose(y_original, y_compiled), 
            f"Device {device_str}: Mismatch between eager and compiled modes. Max diff: {diff}"
        )

    def test_cpu(self):
        self._test_device('cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda(self):
        self._test_device('cuda')

if __name__ == '__main__':
    unittest.main()