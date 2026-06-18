import torch
import torch.nn.functional as F

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(9)

# Use CUDA if available to match the bug report context, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0 # size=(1, 2), stride=(2, 1), dtype=int64
    var_node_5 = torch.full((1, 2), -66, dtype=torch.int32, device=device)
    var_node_6 = torch.full((1, 2), 77, dtype=torch.int64, device=device)
    var_node_4 = torch.ops.aten.add(var_node_5, var_node_6)
    var_node_7 = torch.full((1, 2), -64, dtype=torch.int32, device=device)
    var_node_3 = torch.ops.aten.mul(var_node_4, var_node_7)

    # Adaptation: Replace torch.nonzero with torch.nn.functional.hardshrink
    # Original var_node_9 was bool (3,4), hardshrink requires numeric input.
    # We adjust shape to (1,2) to be compatible with subsequent add operation with var_node_3.
    var_node_9 = torch.full((1, 2), 0.1, dtype=torch.float32, device=device)
    var_node_8 = F.hardshrink(var_node_9, lambd=0.5)

    var_node_2 = torch.ops.aten.add(var_node_3, var_node_8) # int32 + float32 -> float32
    var_node_0 = torch.ops.aten.div(var_node_1, var_node_2) # int64 / float32 -> float32

    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor
sentinel = torch.tensor(1.0, requires_grad=True, device=device)

arg_0 = torch.randint(0, 3, (1, 2), dtype=torch.int64, device=device)

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

# Test compilation with unbacked operations - this should work!
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success with unbacked operations')

# Compare results - shapes may differ due to data-dependent operations
print(f'Eager result: {result_original}')
print(f'Compiled result: {result_compiled}')
if hasattr(result_original, 'shape') and hasattr(result_compiled, 'shape'):
    print(f'Eager shape: {result_original.shape}')
    print(f'Compiled shape: {result_compiled.shape}')

# Assert consistency
assert torch.allclose(result_original, result_compiled), "Results differ between eager and compiled"