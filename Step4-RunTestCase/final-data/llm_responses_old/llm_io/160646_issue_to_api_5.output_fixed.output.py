import tempfile
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Handle missing triton dependency gracefully
try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
except ImportError:
    print("Test Skipped: 'triton' module is not installed.")
    import sys
    sys.exit(0)


# --- Original Bug Reproduction Logic (Triton Op) ---

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


# --- Leveraging Similar API (CustomFromMask) ---

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Add a parameter to apply pruning to, leveraging the CustomFromMask pattern
        self.linear = nn.Linear(1024, 1024)

    def forward(self, q):
        # Pass through the pruned layer
        x = self.linear(q)
        # Call the Triton op
        return my_triton_op(x)


def main():
    model = MyModule().to("cuda")
    
    # Apply CustomFromMask to the linear layer's weight
    # This mirrors the usage pattern of the similar API provided in the issue context
    mask = torch.ones_like(model.linear.weight)
    prune.CustomFromMask.apply(model.linear, 'weight', mask)

    with torch.inference_mode():
        q = torch.randn(1024, device="cuda")
        
        # Export the model
        exported_model = torch.export.export(
            model, 
            (q,), 
            dynamic_shapes=({0: torch.export.Dim("dim")},)
        )
        
    # Attempt AOT compilation which is the failing operation in the bug report
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            torch._inductor.aoti_compile_and_package(
                exported_model,
                package_path=tmpdir + "/package.pt2",
            )
            print("Test Passed: AOT compilation succeeded with Triton op and CustomFromMask.")
        except Exception as e:
            print(f"Test Failed: {e}")
            raise


if __name__ == "__main__":
    main()