import torch
import unittest

class TestTorchProdCompile(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "requires GPU")
    def test_prod_to_cpu(self):
        # Adapted from the original test_triton_kernel_to_cpu
        # Replaces the user-defined Triton kernel with torch.prod
        def f(x):
            # Original: out = torch.zeros_like(x); add_kernel[...](x, y, out, ...)
            # Adapted: Use torch.prod to generate the output
            out = torch.prod(x)
            
            # Original: out_cpu = out.cpu() + 1
            # This pattern (GPU op -> .cpu() -> op) is the focus of the correctness check
            out_cpu = out.cpu() + 1
            return out_cpu

        # Setup inputs on GPU
        x = torch.randn(4, 4, device="cuda")
        
        # Run eager mode
        eager_out = f(x)
        
        # Run compiled mode
        compiled_out = torch.compile(f)(x)
        
        # Verify correctness
        self.assertEqual(compiled_out, eager_out)

if __name__ == "__main__":
    unittest.main()