import torch
from torch.library import Library

# Handle PyTorch version compatibility for impl_abstract
# impl_abstract is available in PyTorch 2.0+
try:
    from torch.library import impl_abstract
    HAS_IMPL_ABSTRACT = True
except ImportError:
    HAS_IMPL_ABSTRACT = False

# Define a custom library to encapsulate the logic from the bug report.
# The bug involves an int64 iota (arange) followed by a max operation.
my_lib = Library("bug_164465_lib", "DEF")

# Define a custom operator that mimics the failing pattern.
my_lib.define("int64_iota_max(int size) -> Tensor")

# Define the abstract implementation (meta kernel) function.
def int64_iota_max_meta(size):
    # The logic is: create an int64 range of 'size', then take the max.
    # Input: int size
    # Output: scalar tensor (0-d) with dtype int64
    return torch.empty((), dtype=torch.int64)

# Register the abstract implementation
if HAS_IMPL_ABSTRACT:
    # Use the target API: torch.library.impl_abstract
    # to register the abstract implementation (meta kernel).
    # This defines the behavior of the operator on FakeTensors (no data).
    impl_abstract("bug_164465_lib::int64_iota_max")(int64_iota_max_meta)
else:
    # Fallback for older PyTorch versions: try to register with "Meta" key
    # This is supported in PyTorch 1.13+
    try:
        my_lib.impl("int64_iota_max", int64_iota_max_meta, "Meta")
    except Exception:
        # If Meta registration fails (older versions), we just skip it.
        # The abstract test will be skipped later.
        pass

# Define the concrete implementation for the operator.
def int64_iota_max_impl(size):
    # Reproduce the sequence from the bug report:
    # iota = torch.ops.prims.iota.default(size, ..., dtype=torch.int64, ...)
    # max_1 = torch.ops.aten.max.default(iota)
    # We use torch.arange as a concrete proxy for prims.iota.
    t = torch.arange(size, dtype=torch.int64, device='cuda')
    return torch.max(t)

# Register the concrete implementation for CUDA.
my_lib.impl("int64_iota_max", int64_iota_max_impl, "CUDA")

# Test Case
if torch.cuda.is_available():
    # 1. Verify the concrete implementation runs correctly.
    result = torch.ops.bug_164465_lib.int64_iota_max(36)
    assert result.dtype == torch.int64
    assert result.item() == 35, f"Expected 35, got {result.item()}"
    print("Concrete execution test passed.")

    # 2. Verify the abstract implementation (meta kernel) is correct.
    # This is crucial for torch.compile, which relies on FakeTensors.
    if HAS_IMPL_ABSTRACT:
        from torch._subclasses import FakeTensorMode
        with FakeTensorMode():
            fake_result = torch.ops.bug_164465_lib.int64_iota_max(36)
            assert fake_result.shape == (), f"Expected shape (), got {fake_result.shape}"
            assert fake_result.dtype == torch.int64, f"Expected dtype int64, got {fake_result.dtype}"
        print("Abstract implementation test passed.")
    else:
        print("Abstract implementation test skipped (requires PyTorch 2.0+).")
else:
    print("CUDA not available, skipping test.")