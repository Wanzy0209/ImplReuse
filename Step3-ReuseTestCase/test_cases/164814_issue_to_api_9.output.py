import torch
from torch.library import Library

# Setup configurations from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch.manual_seed(1000560)

# --- Leverage the Similar API: torch.library.register_kernel ---
# We define a custom library and operator to encapsulate the logic that triggers the bug.
# This allows us to test the behavior of the compilation process with respect to 
# custom kernel implementations involving 0-dimensional tensors.

lib = Library("test_divergence_lib", "DEF")

# Define a custom operation that mimics the transformation in the bug report
# (Tensor of size (1,) -> Tensor of size ())
lib.define("custom_squeeze_to_scalar(Tensor x) -> Tensor")

# The kernel implementation uses as_strided (Original API Under Test) 
# to explicitly perform the dimensionality reduction.
def custom_squeeze_to_scalar_impl(x):
    # x is expected to be size (1,)
    # We manually construct a 0-d tensor view using as_strided.
    # This mirrors the internal behavior that might be failing in the compiler.
    return x.as_strided([], [])

# Register the kernel for CPU (and CUDA if available)
lib.impl("custom_squeeze_to_scalar", custom_squeeze_to_scalar_impl, "CPU")
if torch.cuda.is_available():
    lib.impl("custom_squeeze_to_scalar", custom_squeeze_to_scalar_impl, "CUDA")

# --- Test Case ---

def fuzzed_program(sentinel):
    # Setup inputs matching the original bug report
    var_node_3 = torch.full((2, 3), 3, dtype=torch.int32, device=sentinel.device)
    _inp_unique_wide = torch.arange(1, device=var_node_3.device, dtype=torch.int64)
    _uniq_wide = torch.unique(_inp_unique_wide)
    var_node_2 = _uniq_wide.to(var_node_3.dtype) # size=(1,), stride=(1,), dtype=int32
    
    # Use the registered custom kernel instead of standard reshape/squeeze
    # This tests if the compiler handles the 0-d output of a custom kernel correctly
    var_node_0 = torch.ops.test_divergence_lib.custom_squeeze_to_scalar(var_node_2)
    
    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Test Eager Mode
try:
    result_original = fuzzed_program(sentinel)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Test Compiled Mode
try:
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(sentinel)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')