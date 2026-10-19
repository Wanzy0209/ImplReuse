import torch
import unittest
import gc
import psutil
import numpy as np

def mem_mb_print():
    gc.collect()
    mem = psutil.virtual_memory()
    process = psutil.Process().memory_info().rss

    # Replacement: torch.is_tensor -> torch.from_numpy
    # We wrap the call in a try-except block to ensure the test remains runnable
    # when gc.get_objects() returns items that are not numpy arrays.
    def is_convertible_to_tensor(o):
        try:
            torch.from_numpy(o)
            return True
        except (TypeError, ValueError):
            return False

    count = sum(1 for o in gc.get_objects() if is_convertible_to_tensor(o))

    print(f"used: {mem.used // 1024**1} KB,"
          f"process: {process // 1024**1} KB,"
          f"percent: {mem.percent} %,"
          f"PyTorch tensors (via from_numpy check): {count}")

class TestLeafGradMaxCUDA(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cross_grad_max_cuda(self):
        # Replacement: torch.device -> torch.diff
        # Original: device = torch.device("cuda")
        # We replace the call to torch.device with torch.diff.
        # The result is a tensor, but we keep the variable name to maintain structure.
        device = torch.diff(torch.tensor([1.0, 2.0], device="cuda"))

        # Replacement: torch.randn -> torch.var_mean
        # Original: leaf = torch.randn(8, 1, 24, device=device, requires_grad=True)
        # We replace torch.randn with torch.var_mean. To maintain the shape (8, 1, 24)
        # required for the slicing logic in the loop, we compute variance along the last dimension
        # of a (8, 1, 24, 2) tensor.
        input_data = torch.ones(8, 1, 24, 2, device="cuda", requires_grad=True)
        leaf = torch.var_mean(input_data, dim=3)[0]

        no_leaf = leaf * 1.0

        for _ in range(100):
            window_left = no_leaf[:, :, :-1]          # [8, 1, 23]
            # inplace operators
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