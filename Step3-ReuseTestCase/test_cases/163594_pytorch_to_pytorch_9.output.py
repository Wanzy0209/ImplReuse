import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

class TestBroadcastObjectList(unittest.TestCase):
    """
    Test case for torch.distributed.broadcast_object_list.
    This test is derived from a flaky test involving torch.compile and DTensor redistribution
    that resulted in timeouts. This test verifies that the underlying collective communication
    primitive completes successfully without hanging.
    """

    def setUp(self):
        self.world_size = 2

    def tearDown(self):
        pass

    def _run_broadcast_test(self, rank, world_size):
        # Initialize process group for multiprocessing
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Use 'gloo' backend for CPU testing, which is standard for these types of checks
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # Prepare data: Rank 0 has the data, others have None
        if rank == 0:
            object_list = [
                42, 
                "test_string", 
                {"key": torch.tensor([1.0, 2.0, 3.0])}
            ]
        else:
            object_list = [None, None, None]

        # Call the similar API: torch.distributed.broadcast_object_list
        # This operation must complete to avoid the timeout issues seen in the original bug.
        try:
            dist.broadcast_object_list(object_list, src=0)
        except Exception as e:
            self.fail(f"broadcast_object_list failed on rank {rank}: {e}")

        # Verify the broadcasted data matches the source
        self.assertEqual(object_list[0], 42)
        self.assertEqual(object_list[1], "test_string")
        self.assertTrue(torch.equal(object_list[2]["key"], torch.tensor([1.0, 2.0, 3.0])))

        dist.destroy_process_group()

    def test_broadcast_object_list_timeout_check(self):
        """
        Verifies that broadcast_object_list completes synchronization across processes
        without timing out, addressing the flakiness observed in the original issue.
        """
        if not torch.distributed.is_available():
            self.skipTest("Distributed package not available")

        # Spawn processes to simulate the distributed environment
        mp.spawn(self._run_broadcast_test, args=(self.world_size,), nprocs=self.world_size, join=True)

if __name__ == '__main__':
    unittest.main()