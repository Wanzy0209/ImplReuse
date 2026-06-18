import torch

# Configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(9)

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0 # size=(1, 2), stride=(2, 1), dtype=int64
    
    # Setup for var_node_3
    var_node_5 = torch.full((1, 2), -66, dtype=torch.int32)
    var_node_6 = torch.full((1, 2), 77, dtype=torch.int64)
    var_node_4 = torch.ops.aten.add(var_node_5, var_node_6)
    var_node_7 = torch.full((1, 2), -64, dtype=torch.int32)
    var_node_3 = torch.ops.aten.mul(var_node_4, var_node_7)

    # --- Adapted Call Site ---
    # Original API: torch.nonzero(var_node_9)
    # Similar API: torch.addmm
    # We adapt the inputs to produce a tensor of shape (1, 2) to match the subsequent operations.
    # addmm(input, mat1, mat2) -> (1, 2)
    # input: (1, 2), mat1: (1, 3), mat2: (3, 2)
    addmm_input = torch.full((1, 2), 1, dtype=torch.int64)
    addmm_mat1 = torch.full((1, 3), 2, dtype=torch.int64)
    addmm_mat2 = torch.full((3, 2), 3, dtype=torch.int64)
    
    var_node_8 = torch.addmm(addmm_input, addmm_mat1, addmm_mat2)
    # -------------------------

    var_node_2 = torch.ops.aten.add(var_node_3, var_node_8)
    var_node_0 = torch.ops.aten.div(var_node_1, var_node_2)
    
    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.randint(0, 3, (1, 2), dtype=torch.int64)

args = (arg_0,) + (sentinel,)

# Test Eager Execution
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')
    raise

# Test Compilation
try:
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')
    raise

# Compare results
print(f'Eager result: {result_original}')
print(f'Compiled result: {result_compiled}')

if hasattr(result_original, 'shape') and hasattr(result_compiled, 'shape'):
    print(f'Eager shape: {result_original.shape}')
    print(f'Compiled shape: {result_compiled.shape}')

# Assert equality to check for divergence
assert torch.allclose(result_original, result_compiled), "Divergence detected between eager and compiled results!"