import torch
import torch.nn.functional as F

# Check if torch._dynamo is available (PyTorch 2.0+)
has_dynamo = hasattr(torch, '_dynamo')

if has_dynamo:
    # Configuration from the original bug report to trigger the specific compilation path
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
else:
    print("Warning: torch._dynamo is not available. Skipping torch._dynamo specific configuration and compilation test.")

torch.manual_seed(9)

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0 # size=(1, 2), stride=(2, 1), dtype=int64, device=cuda (if available)
    var_node_5 = torch.full((1, 2), -66, dtype=torch.int32) # size=(1, 2), stride=(2, 1), dtype=int32
    var_node_6 = torch.full((1, 2), 77, dtype=torch.int64) # size=(1, 2), stride=(2, 1), dtype=int64
    var_node_4 = torch.ops.aten.add(var_node_5, var_node_6) # size=(1, 2), stride=(2, 1), dtype=int32
    var_node_7 = torch.full((1, 2), -64, dtype=torch.int32) # size=(1, 2), stride=(2, 1), dtype=int32
    var_node_3 = torch.ops.aten.mul(var_node_4, var_node_7) # size=(1, 2), stride=(2, 1), dtype=int32
    
    # Adaptation: Replace torch.nonzero with torch.nn.functional.tanhshrink
    # Original: var_node_9 = torch.full((3, 4), False, dtype=torch.bool)
    # Original: var_node_8 = torch.nonzero(var_node_9) -> size=(1, 2)
    # To maintain shape compatibility for the subsequent add operation with var_node_3 (size 1, 2),
    # we adjust var_node_9 to be size (1, 2). tanhshrink preserves shape.
    var_node_9 = torch.full((1, 2), 0.5, dtype=torch.float32) 
    var_node_8 = F.tanhshrink(var_node_9) # size=(1, 2), dtype=float32
    
    var_node_2 = torch.ops.aten.add(var_node_3, var_node_8) # size=(1, 2), dtype=float32 (promotion)
    var_node_0 = torch.ops.aten.div(var_node_1, var_node_2) # size=(1, 2), dtype=float32
    
    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Input argument
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
result_compiled = None
if has_dynamo:
    try:
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        result_compiled = compiled_program(*args)
        print(' compile success with unbacked operations')
    except Exception as e:
        print(f' compile failed: {e}')
        raise
else:
    print("Skipping compilation test as torch._dynamo is not available.")

# Compare results
if has_dynamo and result_compiled is not None:
    print(f'Eager result: {result_original}')
    print(f'Compiled result: {result_compiled}')
    if hasattr(result_original, 'shape') and hasattr(result_compiled, 'shape'):
        print(f'Eager shape: {result_original.shape}')
        print(f'Compiled shape: {result_compiled.shape}')

    # Assertion to check for divergence (the core issue being tested)
    assert torch.allclose(result_original, result_compiled), "Divergence detected between eager and compiled modes"
else:
    print("Skipping result comparison as compiled result is not available.")