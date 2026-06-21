import torch
import sys

# Check if torch._dynamo is available (requires PyTorch 2.0+)
if not hasattr(torch, '_dynamo'):
    print("Skipping test: torch._dynamo is not available (requires PyTorch 2.0+)")
    sys.exit(0)

# Reproduce the configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(13653)

def fuzzed_program(arg_0, arg_1, sentinel):
    # Preserve the complex arithmetic logic that triggered the original compile error
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
    var_node_0 = var_node_1 * var_node_7
    
    # Calculate the intermediate result
    result = var_node_0 * sentinel
    
    # Adaptation: Use torch.any on the resulting tensor
    # This tests the similar API (torch.any) within the context of the bug
    return torch.any(result)

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
arg_1 = torch.tensor(torch.randn(()), dtype=torch.int64).item()

args = (arg_0, arg_1, sentinel)

# Run eager mode
result_original = fuzzed_program(*args)
print(' eager success')

# Run compiled mode
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')

# Verify results match
assert result_original == result_compiled, f"Eager and compiled results differ: {result_original} vs {result_compiled}"