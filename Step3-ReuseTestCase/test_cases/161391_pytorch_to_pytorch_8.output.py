import torch
import torch.nn.functional as F
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

class TestLeafGradCosineSimilarityCUDA(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cross_grad_cosine_similarity_cuda(self):
        device = torch.device("cuda")

        # leaf GPU tensor
        leaf = torch.randn(8, 1, 24, device=device, requires_grad=True)
        no_leaf = leaf * 1.0

        for _ in range(100):
            # Prepare slices for cosine similarity
            # x1 and x2 are views of the non-leaf tensor
            x1 = no_leaf[:, :, :-1]  # [8, 1, 23]
            x2 = no_leaf[:, :, 1:]   # [8, 1, 23]
            
            # Calculate cosine similarity
            # Result is [8, 1] (reduced along dim 2)
            sim = F.cosine_similarity(x1, x2, dim=2)
            
            # In-place assignment: assign the result back to a slice of the non-leaf tensor
            # This mimics the in-place operation pattern from the original bug report
            no_leaf[:, :, 0] = sim
            
            # Python arithmetic operators
            mask = (no_leaf >= 0)
            mem_mb_print()

        loss = mask.sum()

if __name__ == "__main__":
    unittest.main()