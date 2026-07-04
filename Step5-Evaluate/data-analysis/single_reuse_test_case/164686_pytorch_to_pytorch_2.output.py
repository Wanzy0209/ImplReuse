import torch

# Configuration from the original bug report
# Add a check to handle environments where torch._dynamo is not available
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
else:
    print("Warning: torch._dynamo is not available. Skipping dynamo-specific configuration.")

torch.manual_seed(13653)

def fuzzed_program(arg_0, arg_1, sentinel):
    # Replicating the scalar arithmetic logic from the original bug
    var_node_3 = torch.full((), 1.0, dtype=torch.float32)
    var_node_2 = var_node_3.item()
    var_node_5 = -3
    var_node_6 = arg_0
    var_node_4 = var_node_5 + var_node_6
    var_node_1 = var_node_2 + var_node_4

    var_node_9 = 1
    var_node_10 = -10
    var_node_8 = var_node_9 / var_node_10
    
    var_node_12 = arg_1
    var_node_13 = -5
    var_node_11 = var_node_12 / var_node_13
    
    var_node_7 = var_node_8 + var_node_11

    # Adaptation: Replace direct multiplication with torch.prod
    # We construct a tensor from the computed scalars and apply torch.prod
    input_tensor = torch.tensor([var_node_1, var_node_7])
    var_node_0 = torch.prod(input_tensor)

    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Inputs matching the original bug report
arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
arg_1 = torch.tensor(torch.randn(()), dtype=torch.int64).item()

args = (arg_0, arg_1) + (sentinel,)

# Test Eager mode
result_original = fuzzed_program(*args)
print(' eager success')

# Test Compiled mode
# Only run compilation if torch._dynamo is available
if hasattr(torch, '_dynamo'):
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')

    # Verify results match
    assert torch.allclose(result_original, result_compiled), "Results differ between eager and compiled modes"
else:
    print("Skipping compiled mode test as torch._dynamo is not available.")