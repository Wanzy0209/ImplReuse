import torch
import unittest
import gc
import psutil

def mem_mb_print():
    gc.collect()
    process = psutil.Process().memory_info().rss
    print(f"Process Memory: {process // 1024**2} MB")

class TestMemoryLeakMax(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_inplace_max_memory_leak(self):
        device = torch.device("cuda")

        # leaf GPU tensor
        # Note: The suggested replacement 'torch.nn.functional.cosine_embedding_loss' 
        # is a loss function and cannot be used to generate random input tensors.
        # We retain torch.randn to ensure the test is runnable and valid.
        leaf = torch.randn(8, 1, 24, device=device, requires_grad=True)
        no_leaf = leaf * 1.0

        for _ in range(100):
            window_left = no_leaf[:, :, :-1]          # [8, 1, 23]
            # inplace operators
            # The API under test is torch.max (same as original)
            no_leaf[:, :, 1:] = torch.max(
                no_leaf[:, :, 1:],  
                window_left   
            )
            # Python arithmetic operators
            mask = (no_leaf >= 0)
            mem_mb_print()

        loss = mask.sum()

if __name__ == "__main__":
    unittest.main()