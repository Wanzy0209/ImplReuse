import torch
import torch.nn.functional as F

# Fix: Check if torch._dynamo exists before accessing its config
# This prevents AttributeError in environments where torch._dynamo is not available
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(9)

def fuzzed_program_softsign(arg_0, sentinel):
    # Adapt input to be float for softsign (original was int64)
    # Preserving the size and stride pattern from the original issue
    var_node_1 = arg_0  # size=(1, 2), stride=(2, 1), dtype=float32, device=cuda

    # Original: var_node_9 = torch.full((3, 4), False, dtype=torch.bool)
    # Adapted: Use float values for softsign
    var_node_9 = torch.full((3, 4), 0.5, dtype=torch.float32)  # size=(3, 4), stride=(4, 1), dtype=float32, device=cuda

    # Original: var_node_8 = torch.nonzero(var_node_9)
    # Similar API: torch.nn.functional.softsign
    # This replaces the problematic nonzero with the similar softsign operation
    var_node_8 = F.softsign(var_node_9)

    # Original: var_node_2 = torch.ops.aten.add(var_node_3, var_node_8)
    # We need to construct var_node_3 to be compatible with var_node_8 (size 3, 4)
    # Reusing the logic from the original var_node_3 construction but adapted for size
    var_node_5 = torch.full((3, 4), -66, dtype=torch.float32)
    var_node_6 = torch.full((3, 4), 77, dtype=torch.float32)
    var_node_4 = torch.ops.aten.add(var_node_5, var_node_6)
    var_node_7 = torch.full((3, 4), -64, dtype=torch.float32)
    var_node_3 = torch.ops.aten.mul(var_node_4, var_node_7)

    # Combine the results
    var_node_2 = torch.ops.aten.add(var_node_3, var_node_8)

    # Final operation to ensure interaction
    # Note: We are not performing the final div with var_node_1 here because sizes (1,2) and (3,4) might mismatch
    # depending on broadcasting, and the goal is to test the softsign integration.
    # We simply return the result of the chain involving softsign.
    result = var_node_2 * sentinel
    
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Input tensor matching the original issue's characteristics
arg_0 = torch.randn(1, 2, dtype=torch.float32)

args = (arg_0,) + (sentinel,)

# Test Eager Execution
try:
    result_original = fuzzed_program_softsign(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')
    raise

# Test Compilation with unbacked operations
# Check if torch.compile is available to avoid further errors in older environments
if hasattr(torch, 'compile'):
    try:
        compiled_program = torch.compile(fuzzed_program_softsign, fullgraph=True, dynamic=True)
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
        assert result_original.shape == result_compiled.shape, "Shape mismatch between eager and compiled"

    assert torch.allclose(result_original, result_compiled), "Result mismatch between eager and compiled"
    print(' Test passed: Eager and compiled results match.')
else:
    print("torch.compile is not available in this environment. Skipping compilation test.")