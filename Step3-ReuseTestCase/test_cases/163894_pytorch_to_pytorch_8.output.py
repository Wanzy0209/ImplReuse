import torch

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(9)

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0 # size=(1, 2), stride=(2, 1), dtype=int64
    var_node_5 = torch.full((1, 2), -66, dtype=torch.int32) # size=(1, 2), stride=(2, 1), dtype=int32
    var_node_6 = torch.full((1, 2), 77, dtype=torch.int64) # size=(1, 2), stride=(2, 1), dtype=int64
    var_node_4 = torch.ops.aten.add(var_node_5, var_node_6) # size=(1, 2), stride=(2, 1), dtype=int32
    var_node_7 = torch.full((1, 2), -64, dtype=torch.int32) # size=(1, 2), stride=(2, 1), dtype=int32
    var_node_3 = torch.ops.aten.mul(var_node_4, var_node_7) # size=(1, 2), stride=(2, 1), dtype=int32
    
    # --- Adapted Call Site ---
    # Replacing torch.nonzero with torch.addmv
    # addmv(input, mat, vec, *, beta=1, alpha=1)
    # We construct inputs from existing nodes to maintain graph complexity and dependencies.
    # mat: (1, 2), vec: (2), input: (1)
    
    # Cast to float for addmv compatibility (matmul usually requires float/complex)
    mat = var_node_3.to(torch.float32) 
    vec = var_node_1.to(torch.float32).squeeze(0)
    input_tensor = var_node_5.to(torch.float32).squeeze(0)
    
    var_node_8 = torch.addmv(input_tensor, mat, vec) # size=(1), stride=(1), dtype=float32
    # ------------------------

    var_node_2 = torch.ops.aten.add(var_node_3, var_node_8) # size=(1, 2), stride=(2, 1), dtype=float32
    var_node_0 = torch.ops.aten.div(var_node_1, var_node_2) # size=(1, 2), stride=(2, 1), dtype=float32
    
    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.randint(0, 3, (1, 2), dtype=torch.int64)

args = (arg_0,) + (sentinel,)

# Test Eager Mode
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')
    raise

# Test Compilation with unbacked operations
try:
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success with unbacked operations')
except Exception as e:
    print(f' compile failed: {e}')
    raise

# Compare results
print(f'Eager result: {result_original}')
print(f'Compiled result: {result_compiled}')

if hasattr(result_original, 'shape') and hasattr(result_compiled, 'shape'):
    print(f'Eager shape: {result_original.shape}')
    print(f'Compiled shape: {result_compiled.shape}')

assert torch.allclose(result_original, result_compiled), "Divergence detected between eager and compiled modes!"