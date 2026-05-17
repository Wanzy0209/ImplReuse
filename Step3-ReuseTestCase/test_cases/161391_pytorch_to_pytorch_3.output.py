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

class TestLeafGradCumsumCUDA(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cross_grad_cumsum_cuda(self):
        device = torch.device("cuda")

        # leaf GPU tensor
        leaf = torch.randn(8, 1, 24, device=device, requires_grad=True)
        no_leaf = leaf * 1.0

        for _ in range(100):
            # Adapted operation: using torch.cumsum instead of torch.max
            # The original bug pattern involves assigning the result of an op to a slice of a non-leaf tensor
            no_leaf[:, :, 1:] = torch.cumsum(no_leaf[:, :, 1:], dim=2)

            # Python arithmetic operators that trigger the graph node loss/leak
            mask = (no_leaf >= 0)
            mem_mb_print()

        loss = mask.sum()

if __name__ == "__main__":
    unittest.main()