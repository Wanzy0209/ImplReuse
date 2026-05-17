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
        
        # Create a leaf tensor with requires_grad=True
        leaf = torch.randn(8, 1, 24, device=device, requires_grad=True)
        # Create a non-leaf tensor
        no_leaf = leaf * 1.0

        print("Starting loop to check for memory leak...")
        for i in range(100):
            window_left = no_leaf[:, :, :-1]
            
            # In-place operation using torch.max
            # This is the call site for the similar API
            no_leaf[:, :, 1:] = torch.max(
                no_leaf[:, :, 1:], 
                window_left
            )
            
            # Python arithmetic operators that might trigger the leak
            mask = (no_leaf >= 0)
            
            if i % 10 == 0:
                mem_mb_print()

        # The test passes if it runs without OOM, 
        # but the bug report indicates memory grows.
        # We keep the print to observe the behavior.
        loss = mask.sum()
        # loss.backward() # Optional, depending on if we want to trigger backward pass

if __name__ == "__main__":
    unittest.main()