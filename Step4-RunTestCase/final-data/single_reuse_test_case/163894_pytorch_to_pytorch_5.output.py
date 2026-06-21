import torch

# Configuration from the original bug report
# Added check for torch._dynamo to handle older PyTorch versions or missing dependencies
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
else:
    print("Warning: torch._dynamo is not available. Skipping dynamo configuration.")

torch.manual_seed(9)

def fuzzed_program(arg_0, sentinel):
    # Setup tensors matching the original structure
    # Note: Added device='cuda' to match the comments in the bug report
    var_node_1 = arg_0
    var_node_5 = torch.full((1, 2), -66, dtype=torch.int32, device='cuda')
    var_node_6 = torch.full((1, 2), 77, dtype=torch.int64, device='cuda')
    var_node_4 = torch.ops.aten.add(var_node_5, var_node_6)
    var_node_7 = torch.full((1, 2), -64, dtype=torch.int32, device='cuda')
    var_node_3 = torch.ops.aten.mul(var_node_4, var_node_7)
    var_node_9 = torch.full((3, 4), False, dtype=torch.bool, device='cuda')

    # --- Adapted API Call ---
    # Original: var_node_8 = torch.nonzero(var_node_9)
    # Similar API: torch.take
    # We create an index tensor to mimic the shape (1, 2) mentioned in the original comments
    # and cast the result to int64 to match the output type of nonzero.
    index_tensor = torch.tensor([[0, 1]], dtype=torch.long, device='cuda')
    var_node_8 = torch.take(var_node_9, index_tensor).to(torch.int64)
    # ------------------------

    var_node_2 = torch.ops.aten.add(var_node_3, var_node_8)
    var_node_0 = torch.ops.aten.div(var_node_1, var_node_2)
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor
sentinel = torch.tensor(1.0, requires_grad=True)
# Input argument
arg_0 = torch.randint(0, 3, (1, 2), dtype=torch.int64, device='cuda')

args = (arg_0,) + (sentinel,)

# Run Eager
result_original = fuzzed_program(*args)
print(' eager success')

# Run Compiled
# Added check for torch.compile as it depends on torch._dynamo
if hasattr(torch, 'compile'):
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success with unbacked operations')

    # Comparison
    print(f'Eager result: {result_original}')
    print(f'Compiled result: {result_compiled}')
    if hasattr(result_original, 'shape') and hasattr(result_compiled, 'shape'):
        print(f'Eager shape: {result_original.shape}')
        print(f'Compiled shape: {result_compiled.shape}')
else:
    print("Skipping compiled execution: torch.compile is not available.")