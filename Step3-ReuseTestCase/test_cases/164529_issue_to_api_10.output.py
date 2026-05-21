import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.distributed as dist
from typing import List, Optional

# Constants for the proposed API
NCCL_SHRINK_DEFAULT = 0
NCCL_SHRINK_ABORT = 1

class ProcessGroupCreator:
    """
    Mimics the tf.compat.v1.train.SessionCreator pattern for managing 
    the lifecycle of a ProcessGroup, including the new shrink capability.
    """
    def __init__(self, backend: str = 'gloo'):
        self.backend = backend
        self.pg = None

    def create_group(self):
        """Initializes the default process group."""
        if not dist.is_initialized():
            # Mocking initialization for unit test context
            dist.init_process_group(
                backend=self.backend, 
                init_method='tcp://127.0.0.1:29500', 
                rank=0, 
                world_size=2
            )
        self.pg = dist.group.WORLD
        return self.pg

    def shrink_and_get_group(self, ranks_to_exclude: List[int], shrink_flags: int = NCCL_SHRINK_DEFAULT):
        """
        Leverages the new shrink_group API to exclude faulty ranks.
        This mirrors the 'create_session' pattern but for reconfiguration.
        """
        # The API call under test
        dist.shrink_group(ranks_to_exclude, self.pg, shrink_flags)
        return self.pg

class TestShrinkGroup(unittest.TestCase):
    """
    Test case for torch.distributed.shrink_group API.
    Preserves the fault tolerance logic (excluding ranks) and uses 
    a Creator pattern similar to tf.compat.v1.train.SessionCreator.
    """

    @patch('torch.distributed.shrink_group')
    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.init_process_group')
    def test_shrink_group_excludes_faulty_ranks(self, mock_init, mock_is_init, mock_shrink):
        """
        Test that shrink_group is called with the correct arguments 
        to exclude faulty ranks, ensuring the group can recover 
        without hanging.
        """
        # Setup: Create a group using the Creator pattern
        creator = ProcessGroupCreator()
        pg = creator.create_group()
        
        # Verify initial setup
        self.assertIsNotNone(pg)

        # Action: Simulate a fault where rank 1 needs to be excluded
        faulty_ranks = [1]
        
        # Call the shrink logic
        # Using NCCL_SHRINK_DEFAULT as per the proposed API
        updated_pg = creator.shrink_and_get_group(faulty_ranks, NCCL_SHRINK_DEFAULT)

        # Assertion: Verify the underlying API was called correctly
        mock_shrink.assert_called_once_with(
            faulty_ranks, 
            pg, 
            NCCL_SHRINK_DEFAULT
        )
        
        # Ensure the process group reference is maintained
        self.assertEqual(updated_pg, pg)

    @patch('torch.distributed.shrink_group')
    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.init_process_group')
    def test_shrink_group_with_abort_flag(self, mock_init, mock_is_init, mock_shrink):
        """
        Test shrink_group with the ABORT flag to terminate ongoing operations.
        """
        creator = ProcessGroupCreator()
        pg = creator.create_group()
        
        faulty_ranks = [0]
        
        # Call with ABORT flag
        creator.shrink_and_get_group(faulty_ranks, NCCL_SHRINK_ABORT)

        # Verify the flag was passed correctly
        mock_shrink.assert_called_once_with(
            faulty_ranks, 
            pg, 
            NCCL_SHRINK_ABORT
        )

if __name__ == '__main__':
    unittest.main()