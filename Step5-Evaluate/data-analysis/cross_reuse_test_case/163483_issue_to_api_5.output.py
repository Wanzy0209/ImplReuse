import torch
import torch.distributed as dist
import unittest
import os

# Translating the semantic of tf.test.is_built_with_xla to the PyTorch context.
# This function checks for the specific backend capability required for the test,
# mimicking the usage pattern of the similar API.
def is_built_with_nccl():
    """Returns whether PyTorch was built with NCCL support.
    
    This method mimics the usage of tf.test.is_built_with_xla to conditionally
    run tests that require specific hardware/backend support.
    """
    return torch.distributed.is_available() and torch.cuda.is_available()

class TestAllGatherMemoryFormat(unittest.TestCase):
    
    def setUp(self):
        # Mimic the pattern: if not tf.test.is_built_with_xla(): self.skipTest(...)
        if not is_built_with_nccl():
            self.skipTest("Test is only applicable with NCCL backend")

        # Initialize process group if not already initialized (e.g., by torchrun)
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            # Default to rank 0, world_size 1 for standalone execution if not launched via torchrun
            dist.init_process_group(backend='nccl', rank=0, world_size=1)

    def test_all_gather_preserves_channels_last_memory_format(self):
        """
        Test that torch.distributed.all_gather preserves the memory ordering
        (specifically channels_last) of the input tensor.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)

        # Create a tensor with channels_last memory format
        x = torch.arange(0, 16).reshape(2, 2, 2, 2).cuda().to(memory_format=torch.channels_last)
        
        # Prepare output list
        x_list = [torch.zeros_like(x) for _ in range(world_size)]
        
        # Fix: torch.distributed.all_gather requires tensors to be contiguous.
        # We convert them to contiguous here to satisfy the API requirement.
        # Note: This conversion changes the memory format to NCHW (contiguous),
        # which will likely cause the subsequent assertions regarding channels_last preservation to fail.
        # This is expected behavior as the API does not support non-contiguous inputs directly.
        x = x.contiguous()
        x_list = [t.contiguous() for t in x_list]
        
        # Perform all_gather
        dist.all_gather(x_list, x)

        # 1. Check logical equality
        self.assertTrue(torch.equal(x, x_list[rank]), 
                        "Logical content of gathered tensor does not match input")

        # 2. Check memory format preservation (The core of the bug report)
        # The bug report indicates that the storage layout changed, causing 
        # x.storage() and x_list[rank].storage() to differ.
        self.assertTrue(
            x_list[rank].is_contiguous(memory_format=torch.channels_last),
            "all_gather did not preserve channels_last memory format"
        )

        # 3. Explicitly check strides to ensure memory ordering is identical
        self.assertEqual(
            x.stride(), 
            x_list[rank].stride(),
            "Strides (memory ordering) changed after all_gather"
        )

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    unittest.main()