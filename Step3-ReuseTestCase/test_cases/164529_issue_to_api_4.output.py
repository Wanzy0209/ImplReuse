import torch
import unittest
from unittest.mock import patch, MagicMock
from typing import List, Optional

# Mocking the torch.distributed module structure for the test
class MockProcessGroup:
    def __init__(self, size, rank):
        self.size_val = size
        self.rank_val = rank

    def size(self):
        return self.size_val

    def rank(self):
        return self.rank_val

# Constants defined in the issue description
NCCL_SHRINK_DEFAULT = 0
NCCL_SHRINK_ABORT = 1

class TestShrinkGroupAPI(unittest.TestCase):
    """
    Test case for the proposed shrink_group API.
    This test leverages the pattern from tf.keras.backend.epsilon, which relies
    on a default internal constant (_EPSILON). Similarly, shrink_group relies on
    a default flag (NCCL_SHRINK_DEFAULT).
    """

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    def test_shrink_group_uses_default_flags(self, mock_get_world_size, mock_get_rank):
        """
        Test that shrink_group correctly applies the default NCCL_SHRINK_DEFAULT flag
        when no explicit flags are provided, mirroring how tf.keras.backend.epsilon
        returns its default constant value.
        """
        # Setup environment
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 4
        
        # Create a mock ProcessGroup
        pg = MockProcessGroup(size=4, rank=0)

        # Define the mock implementation of shrink_group
        # (Since this is an RFC, we mock the function to simulate its intended behavior)
        def mock_shrink_impl(ranks_to_exclude: List[int], 
                             Pg: Optional['MockProcessGroup'] = None, 
                             shrink_flags: int = NCCL_SHRINK_DEFAULT):
            # Reproduction logic: Ensure the function handles the exclusion list
            self.assertIsInstance(ranks_to_exclude, list)
            self.assertIn(1, ranks_to_exclude)
            
            # Similarity logic: Verify the default flag is used, 
            # just as epsilon() returns _EPSILON by default.
            self.assertEqual(shrink_flags, NCCL_SHRINK_DEFAULT, 
                             "shrink_flags should default to NCCL_SHRINK_DEFAULT")
            return True

        with patch('torch.distributed.shrink_group', side_effect=mock_shrink_impl) as mock_shrink:
            # Scenario: Rank 1 is faulty and needs to be excluded
            faulty_ranks = [1]
            
            # Call the API without specifying shrink_flags to test the default behavior
            # This corresponds to calling tf.keras.backend.epsilon() without arguments
            result = mock_shrink(ranks_to_exclude=faulty_ranks, Pg=pg)
            
            # Assertions
            self.assertTrue(result)
            mock_shrink.assert_called_once()
            
            # Verify the call arguments match the expected pattern
            call_args = mock_shrink.call_args
            self.assertEqual(call_args.kwargs['ranks_to_exclude'], faulty_ranks)
            self.assertEqual(call_args.kwargs['Pg'], pg)
            self.assertEqual(call_args.kwargs['shrink_flags'], NCCL_SHRINK_DEFAULT)

if __name__ == '__main__':
    unittest.main()