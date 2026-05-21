import torch
from torch import Tensor
from torch.library import impl_abstract, impl, define

# Define a custom operator that encapsulates the logic from the bug report.
# This allows us to test the torch.library.impl_abstract API.
define("test_ns::complex_cat_op", (Tensor, Tensor), Tensor)

# Register the abstract implementation (FakeTensor behavior) using the requested API.
# This defines how the operator behaves during tracing/meta-derivatives.
@impl_abstract("test_ns::complex_cat_op")
def complex_cat_op_abstract(x, y):
    # Logic adapted from the bug report.
    # In an abstract implementation, we rely on operators that support FakeTensors
    # (like cat, arange, reshape) to determine output properties.
    y2 = torch.cat(
        [
            x[:, 1:],
            y[:, None] + 32 * 2048,
        ],
        dim=1,
    )

    x2 = x[:, 1:, None]
    y3 = y2[:, -1:, None]

    return (
        torch.cat([x2, y3], dim=1)
        + torch.arange(-2048, 0, device=x.device)[None, None, :]
    ).reshape(1, 32 * 2048)

# Register the concrete implementation for actual execution.
@impl("test_ns::complex_cat_op", "CompositeExplicitAutograd")
def complex_cat_op_impl(x, y):
    y2 = torch.cat(
        [
            x[:, 1:],
            y[:, None] + 32 * 2048,
        ],
        dim=1,
    )

    x2 = x[:, 1:, None]
    y3 = y2[:, -1:, None]

    return (
        torch.cat([x2, y3], dim=1)
        + torch.arange(-2048, 0, device=x.device)[None, None, :]
    ).reshape(1, 32 * 2048)

# Test the implementation
if __name__ == "__main__":
    # Use CPU to ensure the test runs in all environments, 
    # though the original bug was CUDA specific.
    device = "cpu"

    # 1. Test the Abstract Implementation directly using FakeTensors.
    # This verifies that the logic provided to torch.library.impl_abstract
    # correctly infers shapes and dtypes without data.
    x_meta = torch.empty(1, 32, dtype=torch.int64, device="meta")
    y_meta = torch.empty(1, dtype=torch.int32, device="meta")

    try:
        out_meta = torch.ops.test_ns.complex_cat_op(x_meta, y_meta)
        assert out_meta.shape == (1, 32 * 2048), f"Shape mismatch: {out_meta.shape}"
        print("Abstract implementation (torch.library.impl_abstract) test passed.")
    except Exception as e:
        print(f"Abstract implementation test failed: {e}")
        raise

    # 2. Test the Concrete Implementation to ensure runtime correctness.
    x_real = torch.zeros(1, 32, dtype=torch.int64, device=device)
    y_real = torch.zeros(1, dtype=torch.int32, device=device)

    try:
        out_real = torch.ops.test_ns.complex_cat_op(x_real, y_real)
        assert out_real.shape == (1, 32 * 2048), f"Shape mismatch: {out_real.shape}"
        print("Concrete implementation test passed.")
    except Exception as e:
        print(f"Concrete implementation test failed: {e}")
        raise