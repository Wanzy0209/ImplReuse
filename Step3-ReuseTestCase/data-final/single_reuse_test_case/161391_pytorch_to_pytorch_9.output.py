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

class TestLeafGradCholeskySolveCUDA(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cross_grad_cholesky_solve_cuda(self):
        device = torch.device("cuda")

        # leaf GPU tensor
        leaf = torch.randn(8, 1, 24, device=device, requires_grad=True)
        no_leaf = leaf * 1.0

        # Prepare inputs for torch.cholesky_solve
        # The slice we will be modifying is no_leaf[:, :, 1:], which has shape [8, 1, 23]
        # cholesky_solve(input2, input1) where input2 is the RHS and input1 is the Cholesky factor.
        # input2 (RHS) shape: [8, 1, 23]
        # input1 (Cholesky factor) shape: [8, 1, 23, 23]
        
        # Create a valid Cholesky factor (lower triangular with positive diagonal)
        A_raw = torch.randn(8, 1, 23, 23, device=device)
        A = torch.tril(A_raw)
        diag_indices = torch.arange(23, device=device)
        A[..., diag_indices, diag_indices] = torch.abs(A[..., diag_indices, diag_indices]) + 0.1
        
        # Create RHS
        B = torch.randn(8, 1, 23, device=device)

        for _ in range(100):
            # inplace operators (assignment to slice)
            # Replacing torch.max with torch.cholesky_solve
            no_leaf[:, :, 1:] = torch.cholesky_solve(
                B,
                A
            )
            # Python arithmetic operators
            mask = (no_leaf >= 0)
            mem_mb_print()

        loss = mask.sum()

if __name__ == "__main__":
    unittest.main()