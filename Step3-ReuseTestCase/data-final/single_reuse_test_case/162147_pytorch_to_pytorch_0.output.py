import torch
import unittest

class TestTorchIndexSelect(unittest.TestCase):
    def test_index_select_weights_from_tape(self):
        """
        Test case adapted from the FastLearnedCellX3 context.
        Verifies torch.index_select behavior for selecting rows from 
        a weight tensor (tape) based on top-k indices.
        """
        # Simulating the weight tape W: [L, out, in]
        L, out_dim, in_dim = 12, 64, 64
        W = torch.randn(L, out_dim, in_dim)
        
        # Simulating top-k indices: [N, k]
        batch_size, k = 4, 3
        # Generate random indices within the valid range [0, L)
        indices = torch.randint(0, L, (batch_size, k))
        
        # Flatten indices to use with index_select (requires 1D index)
        flat_indices = indices.view(-1)
        
        # Perform index selection along dimension 0 (the 'L' dimension)
        selected_W = torch.index_select(W, dim=0, index=flat_indices)
        
        # Reshape back to [N, k, out, in] to match batch structure
        selected_W = selected_W.view(batch_size, k, out_dim, in_dim)
        
        # Assertions
        self.assertEqual(selected_W.shape, (batch_size, k, out_dim, in_dim))
        
        # Verify correctness by comparing with direct indexing
        for b in range(batch_size):
            for i in range(k):
                idx = indices[b, i]
                self.assertTrue(torch.allclose(selected_W[b, i], W[idx]))

    def test_index_select_basic(self):
        """Basic functionality test for torch.index_select."""
        x = torch.arange(10).reshape(5, 2)
        index = torch.tensor([0, 2, 4])
        
        # Select rows
        result = torch.index_select(x, 0, index)
        expected = torch.tensor([[0, 1], [4, 5], [8, 9]])
        self.assertTrue(torch.equal(result, expected))
        
        # Select columns
        index_col = torch.tensor([1])
        result_col = torch.index_select(x, 1, index_col)
        expected_col = torch.tensor([[1], [3], [5], [7], [9]])
        self.assertTrue(torch.equal(result_col, expected_col))

    def test_index_select_empty_index(self):
        """Test edge case with empty index tensor."""
        x = torch.randn(5, 3)
        index = torch.tensor([], dtype=torch.long)
        result = torch.index_select(x, 0, index)
        self.assertEqual(result.shape, (0, 3))

if __name__ == '__main__':
    unittest.main()