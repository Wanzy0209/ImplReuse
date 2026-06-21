import torch
import unittest
import gc

# Handle missing psutil dependency gracefully
try:
    import psutil
except ImportError:
    psutil = None

def mem_mb_print():
    gc.collect()
    if psutil is not None:
        mem = psutil.virtual_memory()
        process = psutil.Process().memory_info().rss
        print(f"used: {mem.used // 1024**1} KB,"
              f"process: {process // 1024**1} KB,"
              f"percent: {mem.percent} %,"
              f"PyTorch tensors: {sum(1 for o in gc.get_objects() if torch.is_tensor(o))}")
    else:
        # Fallback if psutil is missing
        print(f"psutil not available. PyTorch tensors: {sum(1 for o in gc.get_objects() if torch.is_tensor(o))}")

class TestLeafGradCumminCUDA(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cross_grad_cummin_cuda(self):
        device = torch.device("cuda")

        # leaf GPU tensor
        leaf = torch.randn(8, 1, 24, device=device, requires_grad=True)
        no_leaf = leaf * 1.0

        for _ in range(100):
            window_left = no_leaf[:, :, :-1]          # [8, 1, 23]
            # inplace operators adapted for torch.cummin
            # torch.cummin returns a tuple (values, indices), so we select the values [0]
            no_leaf[:, :, 1:] = torch.cummin(
                no_leaf[:, :, 1:],
                dim=2
            )[0]
            # Python arithmetic operators
            mask = (no_leaf >= 0)
            mem_mb_print()

        loss = mask.sum()

if __name__ == "__main__":
    unittest.main()