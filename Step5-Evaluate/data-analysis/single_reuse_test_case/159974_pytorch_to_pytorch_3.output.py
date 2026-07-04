import torch
import torch.library

# Define a custom library and operator to test torch.library.impl_abstract
lib = torch.library.Library("test_xpu_lib", "DEF")
lib.define("custom_addcmul(Tensor x, Tensor y, Tensor z) -> Tensor")

# Use the legacy API: torch.library.register_fake
# This defines the behavior for FakeTensors (used during torch.compile tracing)
# Note: impl_abstract is the newer name, but register_fake is the stable alias in many versions
@torch.library.register_fake("test_xpu_lib::custom_addcmul")
def custom_addcmul_abstract(x, y, z):
    # Abstract implementation logic (shape inference)
    # This mimics the logic of the original bug report's function
    return x + y * z

# Define a concrete implementation for XPU to allow eager execution
@torch.library.impl("test_xpu_lib::custom_addcmul", "XPU")
def custom_addcmul_xpu(x, y, z):
    return x + y * z

# Test function using the custom operator
def custom_func(x, y, z):
    return torch.ops.test_xpu_lib.custom_addcmul(x, y, z)

# Setup tensors on XPU
try:
    x = torch.randn(128).to("xpu")
    y = torch.randn(128).to("xpu")
    z = torch.randn(128).to("xpu")
except RuntimeError as e:
    print(f"XPU not available or error initializing XPU: {e}")
    exit(0)

# Test eager mode
try:
    out_eager = custom_func(x, y, z)
    expected = x + (y * z)
    assert torch.allclose(out_eager, expected), "Eager mode output mismatch"
    print("eager mode passed")
except Exception as e:
    print(f"eager mode failed: {e}")
    exit(1)

# Test torch.compile mode
# This relies on the torch.library.register_fake registered above
try:
    custom_func_compiled = torch.compile(custom_func)
    out_compiled = custom_func_compiled(x, y, z)
    assert torch.allclose(out_compiled, expected), "Compiled mode output mismatch"
    print("torch.compile passed")
except Exception as e:
    print(f"torch.compile failed: {e}")
    exit(1)