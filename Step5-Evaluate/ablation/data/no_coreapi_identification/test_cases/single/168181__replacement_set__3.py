import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.testing._internal.common_utils import TestCase, run_tests, requires_gpu, GPU_TYPE

class TestTritonKernels(TestCase):
    @requires_gpu
    def test_triton_kernel_to_cpu(self):
        # Define a simple Triton kernel for addition
        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

        def f(x, y):
            # Replacement: torch.zeros_like -> torch.index_select
            # Original: out = torch.zeros_like(x)
            # We simulate creating a tensor of the same shape using index_select.
            # We create a source tensor of zeros and select all indices along dim 0.
            src = torch.zeros(4, 4, device=x.device)
            indices = torch.arange(4, device=x.device)
            out = torch.index_select(src, 0, indices)
            
            add_kernel[(1,)](x, y, out, 16, 16)
            out_cpu = out.cpu() + 1
            return out_cpu

        # Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool1d
        # Original: x = torch.randn(4, 4, device=GPU_TYPE)
        # We generate x using adaptive_max_pool1d. 
        # We create a dummy input tensor (using ones) and pool it to the desired shape.
        dummy_input_x = torch.ones(4, 4, 8, device=GPU_TYPE)
        x = F.adaptive_max_pool1d(dummy_input_x, output_size=4).squeeze(-1)

        dummy_input_y = torch.ones(4, 4, 8, device=GPU_TYPE)
        y = F.adaptive_max_pool1d(dummy_input_y, output_size=4).squeeze(-1)

        eager_out = f(x, y)
        compiled_out = torch.compile(f)(x, y)
        self.assertEqual(compiled_out, eager_out)

if __name__ == '__main__':
    run_tests()