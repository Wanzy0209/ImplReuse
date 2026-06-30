import tempfile
import sys

import torch

# Handle missing triton dependency gracefully
try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
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
            # Leveraging the code pattern from the similar API torch.backends.mha.get_fastpath_enabled.
            # The API checks torch.jit.is_scripting() to determine behavior.
            # We include this check to ensure the AOT compilation handles the scripting guard correctly
            # within the context of the Triton op.
            if torch.jit.is_scripting():
                # Logic specific to scripting mode, if any
                pass
            
            return my_triton_op(q)


def test_aot_triton_with_scripting_guard():
    """
    Test case for Issue 160646: AOT compilation of Triton op fails in PyTorch 2.8.
    This test preserves the original bug reproduction logic while incorporating
    the compilation mode check pattern found in torch.backends.mha.get_fastpath_enabled.
    """
    if not HAS_TRITON:
        print("Skipping test: triton module is not installed.")
        return

    model = MyModule().to("cuda")
    
    with torch.inference_mode():
        q = torch.randn(1024, device="cuda")
        # Note: dynamic_shapes syntax in the original bug report uses a tuple of dicts
        exported_model = torch.export.export(
            model, 
            (q,), 
            dynamic_shapes=({0: torch.export.Dim("dim")},)
        )
        
    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = tmpdir + "/package.pt2"
        
        # This call is expected to succeed. In PyTorch 2.8 (at the time of the bug),
        # this would fail. The test asserts that the compilation completes without error.
        try:
            torch._inductor.aoti_compile_and_package(
                exported_model,
                package_path=package_path,
            )
        except Exception as e:
            raise AssertionError(
                f"AOT compilation failed for Triton op with scripting guard. "
                f"This indicates a regression of Issue 160646. Error: {e}"
            )


if __name__ == "__main__":
    test_aot_triton_with_scripting_guard()