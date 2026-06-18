import torch
import unittest

class TestMatmulCPURegression(unittest.TestCase):
    def test_matmul_float16_cpu(self):
        """
        Test case for Issue 162683: torch.matmul regression on CPU.
        Verifies correctness for specific shapes and dtype (float16) reported in the bug.
        """
        device = "cpu"
        dtype = torch.float16
        
        # Shapes from the bug report
        test_shapes = [
            ((1, 12, 10, 64), (1, 12, 64, 10)),
            ((1, 12, 10, 10), (1, 12, 10, 64)),
        ]

        for shape_a, shape_b in test_shapes:
            with self.subTest(shape_a=shape_a, shape_b=shape_b):
                # Create tensors matching the original test case generation logic
                A = torch.empty(shape_a, dtype=dtype, device=device).uniform_(0, 1) * 2 - 1
                B = torch.empty(shape_b, dtype=dtype, device=device).uniform_(0, 1) * 2 - 1

                # Perform matmul
                C = torch.matmul(A, B)

                # Expected output shape calculation
                # Batch dimensions (1, 12) are preserved.
                # Matrix mult: (..., M, K) @ (..., K, N) -> (..., M, N)
                expected_shape = (shape_a[0], shape_a[1], shape_a[2], shape_b[3])
                
                self.assertEqual(C.shape, expected_shape)
                self.assertEqual(C.dtype, dtype)
                self.assertFalse(torch.isnan(C).any(), "Output contains NaNs")

if __name__ == "__main__":
    unittest.main()