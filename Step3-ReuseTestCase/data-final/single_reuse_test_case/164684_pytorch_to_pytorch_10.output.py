import torch

# Configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch.manual_seed(19990)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = arg_0
    var_node_1 = torch.squeeze(var_node_2)
    var_node_0 = var_node_1.item() # dtype=bool

    # Adaptation: Use torch.rand with the scalar as a size argument
    # to verify torch.rand's behavior with SymBool in compiled mode.
    # We multiply by sentinel to maintain the gradient graph context.
    rand_tensor = torch.rand(var_node_0)
    result = rand_tensor * sentinel

    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.randint(0, 2, (1,), dtype=torch.bool) > 0

args = (arg_0,) + (sentinel,)

# Eager execution
result_original = fuzzed_program(*args)
print(' eager success')

# Compiled execution
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')

# Verify that the shapes match between eager and compiled modes
assert result_original.shape == result_compiled.shape, f"Shape mismatch: eager {result_original.shape} vs compiled {result_compiled.shape}"