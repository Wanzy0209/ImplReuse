import sys
import tempfile

import torch
import torch.nn.functional as F

# Check for triton availability to handle missing dependencies gracefully
try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
except ImportError:
    print("Triton is not available. Skipping test.")
    sys.exit(0)


@triton.jit
def my_kernel(
    out1,
    out2,
):
    # the actual kernel does something more meaningful of course
    tl.store(out1, 1.0)
    tl.store(out2, 1.0)


@triton_op("repro::my_triton_op", mutates_args={})
def my_triton_op(
    q: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    out1 = torch.empty((1,), device=q.device)
    out2 = torch.empty((1, (q.size(0) + block_size - 1) // block_size), device=q.device)

    wrap_triton(my_kernel)[(1, 1)](
        out1,
        out2,
    )

    return out1


class MyModule(torch.nn.Module):
    def forward(self, q):
        # Preserve original bug reproduction logic: call the custom Triton op
        out = my_triton_op(q)
        
        # Leverage the similar API: torch.nn.functional.rrelu_
        # We apply it to the output tensor to test interaction with in-place ops
        # without mutating the graph inputs directly, which could cause export errors.
        F.rrelu_(out)
        
        return out


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    model = MyModule().to("cuda")
    with torch.inference_mode():
        q = torch.randn(1024, device="cuda")
        # Use dynamic shapes as in the original issue
        exported_model = torch.export.export(
            model, 
            (q,), 
            dynamic_shapes=({0: torch.export.Dim("dim")},)
        )
        
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            torch._inductor.aoti_compile_and_package(
                exported_model,
                package_path=tmpdir + "/package.pt2",
            )
            print("Test Passed: AOT compilation succeeded.")
        except Exception as e:
            print(f"Test Failed: AOT compilation failed with error: {e}")
            raise


if __name__ == "__main__":
    main()