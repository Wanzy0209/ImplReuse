import tempfile

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


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
    def __init__(self):
        super().__init__()

    def forward(self, q):
        return my_triton_op(q)


def main():
    model = MyModule().to("cuda")
    with torch.inference_mode():
        q = torch.randn(1024, device="cuda")
        exported_model = torch.export.export(model, (q,), dynamic_shapes=({0: torch.export.Dim("dim")},))
    with tempfile.TemporaryDirectory() as tmpdir:
        torch._inductor.aoti_compile_and_package(
            exported_model,
            package_path=tmpdir + "/package.pt2",
        )


if __name__ == "__main__":
    main()