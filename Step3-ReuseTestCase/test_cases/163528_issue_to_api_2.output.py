import torch
import torch.nn as nn
import unittest

class SearchSortedMetric(nn.Module):
    """
    A module that uses torch.searchsorted to determine the insertion indices
    of values into a sorted sequence. This mimics the logic of ranking operations
    found in metrics like sparse_top_k_categorical_accuracy.
    """
    def __init__(self, sorted_sequence):
        super().__init__()
        # Register sorted_sequence as a buffer to handle device movement
        self.register_buffer('sorted_sequence', sorted_sequence)

    def forward(self, values):
        # The bug report specifically involves transposition (.T) which affects
        # memory layout and compilation. We preserve this logic.
        # This operation finds where 'values' fit into 'sorted_sequence'.
        return torch.searchsorted(self.sorted_sequence.T, values.T).T

class TestSearchSortedCompile(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, cannot test GPU-specific bug.")

    def test_searchsorted_compile_gpu(self):
        """
        Test that torch.compile works correctly with torch.searchsorted on GPU.
        This test mirrors the logic of checking targets against sorted predictions
        (similar to sparse_top_k_categorical_accuracy) but uses searchsorted.
        """
        # Setup dimensions
        batch_size = 32
        feature_dim = 10
        num_boundaries = 100

        # Generate random data
        torch.manual_seed(42)
        # y_true: values we want to locate (analogous to targets in TF metric)
        y_true = torch.randn(batch_size, feature_dim)
        
        # y_pred_sorted: sorted boundaries (analogous to sorted predictions in TF metric)
        y_pred_sorted = torch.randn(num_boundaries, feature_dim)
        y_pred_sorted = torch.sort(y_pred_sorted, dim=0)[0]

        # Move to GPU
        y_true = y_true.to(self.device)
        y_pred_sorted = y_pred_sorted.to(self.device)

        # Instantiate the model
        model = SearchSortedMetric(y_pred_sorted)
        
        # Compile the model (fullgraph=True is stricter and often exposes bugs)
        compiled_model = torch.compile(model, fullgraph=True)

        # Run eager mode
        with torch.no_grad():
            indices_eager = model(y_true)

        # Run compiled mode
        with torch.no_grad():
            indices_compiled = compiled_model(y_true)

        # Verify that the results match
        # The bug report indicates a mismatch on GPU. This assertion should fail if the bug exists.
        max_diff = torch.max(torch.abs(indices_eager - indices_compiled)).item()
        
        self.assertTrue(
            torch.allclose(indices_eager, indices_compiled),
            f"torch.compile produced different results on GPU. Max diff: {max_diff}"
        )

if __name__ == '__main__':
    unittest.main()