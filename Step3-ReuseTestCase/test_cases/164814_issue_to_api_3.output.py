import torch

# Configuration from the original bug report to trigger the specific code path
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(1000560)

def fuzzed_program(sentinel):
    # Create a base tensor
    var_node_3 = torch.full((2, 3), 3, dtype=torch.int32)
    
    # Create a mask that selects exactly one element.
    # This setup is designed to produce a 1-D tensor of size 1 via masked_select,
    # which will then be squeezed to a 0-D tensor.
    mask = torch.zeros((2, 3), dtype=torch.bool)
    mask[0, 0] = True
    
    # Use the similar API: torch.masked_select
    # This replaces the unique/reshape chain from the original bug with an operation
    # that also deals with dynamic output shapes and underlying stride logic.
    var_node_1 = torch.masked_select(var_node_3, mask)
    
    # Squeeze to get a 0-D tensor (scalar).
    # The original bug reported a "dimensionality of sizes (0) must match 
    # dimensionality of strides (1)" error, likely occurring during this 
    # reduction to scalar in the compiled graph.
    var_node_0 = torch.squeeze(var_node_1)
    
    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

args = (sentinel,)

# Test Eager execution
result_original = fuzzed_program(*args)
print(' eager success')

# Test Compiled execution
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')

# Verify consistency
assert torch.allclose(result_original, result_compiled)