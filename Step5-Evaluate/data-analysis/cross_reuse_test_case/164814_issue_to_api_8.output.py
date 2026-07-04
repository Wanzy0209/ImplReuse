import torch
import sys

# Attempt to import triton dependencies
try:
    import triton
    import triton.language as tl
    from torch.library import wrap_triton, triton_op
except ImportError:
    print("Skipping test: 'triton' module is not installed.")
    sys.exit(0)

# Configure Dynamo settings similar to the bug report to ensure scalar handling is tested
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Define a simple Triton kernel that adds two tensors.
# This kernel is designed to handle 0-dim tensors (scalars) by treating them as 1-element tensors.
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

# Wrap the kernel to make it traceable by torch.compile
wrapped_add = wrap_triton(add_kernel)

# Define the custom operation using the wrapped kernel
@triton_op("test::add_tensors", mutates_args=())
def add_tensors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Calculate grid size. For 0-dim tensors, x.numel() is 1.
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    return wrapped_add(grid, x, y, torch.empty_like(x), BLOCK_SIZE=32)

def fuzzed_program(x, y):
    # The bug report highlights a divergence with 0-dim tensors (size=(), stride=()).
    # We test the wrapped Triton kernel with 0-dim inputs to ensure
    # the metadata handling (sizes/strides) is correct during compilation.
    return add_tensors(x, y)

# Create 0-dim tensors (scalars) to trigger the edge case
# This mimics the state of var_node_0 in the original bug report
x = torch.tensor(5.0)
y = torch.tensor(3.0)

print("Testing wrap_triton with 0-dim tensors (scalars)...")

# Eager execution
result_eager = fuzzed_program(x, y)
print(f' eager success: {result_eager.item()}')

# Compiled execution
# Using fullgraph=True and dynamic=True as in the bug report
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(x, y)
print(f' compile success: {result_compiled.item()}')

# Assert that the results match to catch any divergence
assert torch.allclose(result_eager, result_compiled), \
    f"Divergence detected: eager={result_eager}, compiled={result_compiled}"

print(" Test passed: Eager and compiled results match for 0-dim inputs.")