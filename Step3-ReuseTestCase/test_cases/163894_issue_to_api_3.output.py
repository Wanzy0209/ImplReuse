import torch

# Configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(9)

def fuzzed_program(arg_0, sentinel):
    # Original tensor setup
    var_node_1 = arg_0
    var_node_5 = torch.full((1, 2), -66, dtype=torch.int32, device='cuda')
    var_node_6 = torch.full((1, 2), 77, dtype=torch.int64, device='cuda')
    var_node_4 = torch.ops.aten.add(var_node_5, var_node_6)
    var_node_7 = torch.full((1, 2), -64, dtype=torch.int32, device='cuda')
    var_node_3 = torch.ops.aten.mul(var_node_4, var_node_7)

    # The operation causing the divergence
    var_node_9 = torch.full((3, 4), False, dtype=torch.bool, device='cuda')
    var_node_8 = torch.nonzero(var_node_9)

    # Leveraging the similar API: torch.backends.cudnn.version
    # We integrate this into the computation graph to test interaction
    # with the dynamic shape/stride issues of nonzero.
    cudnn_ver = torch.backends.cudnn.version()
    if cudnn_ver is not None:
        # Create a tensor from the version to perform arithmetic
        version_tensor = torch.tensor(cudnn_ver, dtype=torch.int64, device='cuda')
        # Add version to the result of nonzero
        var_node_8 = torch.ops.aten.add(var_node_8, version_tensor)
    else:
        # Fallback if cuDNN is not available, though the issue implies CUDA context
        var_node_8 = torch.ops.aten.add(var_node_8, torch.tensor(0, dtype=torch.int64, device='cuda'))

    var_node_2 = torch.ops.aten.add(var_node_3, var_node_8)
    var_node_0 = torch.ops.aten.div(var_node_1, var_node_2)

    # Ensure gradient computation
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor
sentinel = torch.tensor(1.0, requires_grad=True)

# Input argument
arg_0 = torch.randint(0, 3, (1, 2), dtype=torch.int64, device='cuda')

args = (arg_0, sentinel)

# Run Eager
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')
    exit(1)

# Run Compiled
try:
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success with unbacked operations')
except Exception as e:
    print(f' compile failed: {e}')
    exit(1)

# Comparison
print(f'Eager result: {result_original}')
print(f'Compiled result: {result_compiled}')
if hasattr(result_original, 'shape') and hasattr(result_compiled, 'shape'):
    print(f'Eager shape: {result_original.shape}')
    print(f'Compiled shape: {result_compiled.shape}')
    assert result_original.shape == result_compiled.shape, "Shape mismatch between eager and compiled"