import tempfile

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


# This test case reproduces the AOT compilation failure for Triton ops
# by implementing a Leaky ReLU operation (similar to torch.nn.functional.leaky_relu_)
# using the Triton pattern that caused the bug (intermediate buffers).

@triton.jit
def leaky_relu_kernel(
    out_ptr,
    temp_ptr,  # Intermediate buffer to mimic the bug report's 'out2'
    in_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)
    
    # Mimic the bug report's logic of writing to an intermediate tensor
    tl.store(temp_ptr + offsets, x, mask=mask)
    
    # Leaky ReLU logic
    y = tl.where(x > 0, x, x * negative_slope)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton_op("test::triton_leaky_relu", mutates_args={})
def triton_leaky_relu(
    x: torch.Tensor,
    negative_slope: float = 0.01,
) -> torch.Tensor:
    out = torch.empty_like(x)
    # Allocate an intermediate tensor that is written to but not returned,
    # mirroring the structure of the failing 'my_triton_op' in the issue.
    temp = torch.empty_like(x)

    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    wrap_triton(leaky_relu_kernel)[grid](
        out,
        temp,
        x,
        n_elements,
        negative_slope,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class LeakyReLUModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return triton_leaky_relu(x)


def main():
    model = LeakyReLUModule().to("cuda")
    
    # Create a dummy input
    x = torch.randn(1024, device="cuda")
    
    # Export the model with dynamic shapes, as done in the original bug report
    with torch.inference_mode():
        exported_model = torch.export.export(
            model, 
            (x,), 
            dynamic_shapes=({0: torch.export.Dim("dim")},)
        )
    
    # Attempt AOT compilation and packaging
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            torch._inductor.aoti_compile_and_package(
                exported_model,
                package_path=tmpdir + "/package.pt2",
            )
            print("Test Passed: AOT compilation successful.")
        except Exception as e:
            print(f"Test Failed: {e}")
            raise


if __name__ == "__main__":
    main()