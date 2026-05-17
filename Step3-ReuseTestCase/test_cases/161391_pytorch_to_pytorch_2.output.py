import torch
import unittest
import gc
import psutil

def mem_mb_print():
    gc.collect()
    mem = psutil.virtual_memory()
    process = psutil.Process().memory_info().rss

    print(f"used: {mem.used // 1024**1} KB,"
          f"process: {process // 1024**1} KB,"
          f"percent: {mem.percent} %,"
          f"PyTorch tensors: {sum(1 for o in gc.get_objects() if torch.is_tensor(o))}")

class TestLeafGradIndexSelectCUDA(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cross_grad_index_select_cuda(self):
        device = torch.device("cuda")

        # leaf GPU tensor
        leaf = torch.randn(8, 1, 24, device=device, requires_grad=True)
        no_leaf = leaf * 1.0

        for _ in range(100):
            window_left = no_leaf[:, :, :-1]          # [8, 1, 23]
            
            # Adapted to use torch.index_select
            # We select indices from window_left to assign back to the slice of no_leaf
            # This mimics the in-place modification pattern of the original bug report
            indices = torch.arange(23, device=device)
            no_leaf[:, :, 1:] = torch.index_select(
                window_left,
                dim=2,
                index=indices
            )
            
            # Python arithmetic operators
            mask = (no_leaf >= 0)
            mem_mb_print()

        loss = mask.sum()

if __name__ == "__main__":
    unittest.main()