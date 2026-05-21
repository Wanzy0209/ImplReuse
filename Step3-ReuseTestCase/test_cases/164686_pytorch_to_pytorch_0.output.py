import torch

# Configuration required to trigger the specific compilation path
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Set seed for reproducibility
torch.manual_seed(13653)

def fuzzed_program(arg_0, arg_1, sentinel):
    # Scalar tensor operations
    var_node_3 = torch.full((), 1.0, dtype=torch.float32)
    var_node_2 = var_node_3.item()

    # Integer arithmetic involving mixed types
    var_node_5 = -3
    var_node_6 = arg_0
    var_node_4 = var_node_5 + var_node_6

    # Float arithmetic
    var_node_1 = var_node_2 + var_node_4

    # Division operations
    var_node_9 = 1
    var_node_10 = -10
    var_node_8 = var_node_9 / var_node_10

    var_node_12 = arg_1
    var_node_13 = -5
    var_node_11 = var_node_12 / var_node_13

    var_node_7 = var_node_8 + var_node_11

    # Final multiplication
    var_node_0 = var_node_1 * var_node_7

    # Gradient computation hook
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Setup inputs
sentinel = torch.tensor(1.0, requires_grad=True)
arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
arg_1 = torch.tensor(torch.randn(()), dtype=torch.int64).item()

args = (arg_0, arg_1, sentinel)

# Run eager mode
result_eager = fuzzed_program(*args)
print(' Eager execution successful')

# Run compiled mode using the similar API (torch.compile)
# Using fullgraph=True and dynamic=True as per the bug report context
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' Compiled execution successful')

# Verify results match
assert torch.allclose(result_eager, result_compiled), "Divergence between eager and compiled results"
print(' Test passed: Eager and compiled results match')