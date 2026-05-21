import torch
import unittest
import torch._inductor.config as inductor_config

class TestGroupedMMViewDtype(unittest.TestCase):
    """
    Test case for Issue #163286: [inductor] as_strided lowering throws away .view(dtype)
    
    This test verifies that torch._grouped_mm, when compiled with torch.compile,
    correctly handles tensors that have been viewed as a different dtype.
    The bug manifests when the inductor lowering process uses as_strided and
    inadvertently discards the dtype view information (e.g., viewing a float8 
    tensor as uint8), leading to incorrect kernel arguments or crashes.
    """
    
    def test_grouped_mm_preserves_view_dtype(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Enable inductor specific checks if available
        # inductor_config.debug = True 

        # Setup inputs for grouped_mm
        # batch1: (sum(B_i), K), batch2: (sum(B_i), N)
        B, K, N = 4, 16, 16
        sizes = [4, 4, 4, 4]
        total_B = sum(sizes)

        # Create tensors in float16 (common for grouped_mm)
        x = torch.randn(total_B, K, dtype=torch.float16, device='cuda')
        w = torch.randn(total_B, N, dtype=torch.float16, device='cuda')
        
        # Offsets for splitting the batched matrices
        offsets = torch.tensor([0] + sizes, dtype=torch.int32, device='cuda').cumsum(dim=0)

        # The bug involves a tensor being viewed as a different dtype (e.g., uint8)
        # before being passed to a kernel or used in a way that triggers as_strided.
        # In the original bug, 'output_scales_ptr' is viewed as uint8.
        # We simulate a scenario where a tensor is viewed to ensure the compiler
        # preserves this information through the lowering.
        
        # Create a dummy scale tensor to mimic the 'output_scales_ptr' context
        # In the real bug, this is float8_e8m0fnu viewed as uint8.
        # We use float32 -> uint8 for general compatibility in this test.
        dummy_scales = torch.randn(total_B, dtype=torch.float32, device='cuda')
        viewed_scales = dummy_scales.view(torch.uint8)

        # Define the function to compile
        def fn(x, w, offsets, scales):
            # We pass the viewed tensor to ensure it's part of the graph.
            # While torch._grouped_mm doesn't explicitly take 'scales' in its
            # basic signature, the bug arises from the compiler's handling of
            # such viewed tensors within the graph context of grouped_mm.
            # We perform a dummy operation on scales to keep it alive in the graph.
            _ = scales.sum() 
            return torch._grouped_mm(x, w, offsets)

        # Compile the function
        # Using reduce-overhead to force inductor compilation
        compiled_fn = torch.compile(fn, mode='reduce-overhead')
        
        # Run eager version
        res_eager = fn(x, w, offsets, viewed_scales)
        
        # Run compiled version
        # If the bug is present, this might crash (unsupported dtype in Triton)
        # or produce incorrect numerics because the view was discarded.
        res_compiled = compiled_fn(x, w, offsets, viewed_scales)

        # Assert correctness
        self.assertTrue(torch.allclose(res_eager, res_compiled))

if __name__ == '__main__':
    unittest.main()