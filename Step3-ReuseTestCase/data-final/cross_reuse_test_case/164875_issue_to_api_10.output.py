import torch
import torch.nn.functional as F

# Reproduce the configuration from the original issue
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(1014698)

def fuzzed_program(arg_0, sentinel):
    # Original API: torch.add(var_node_1, var_node_2)
    # Similar API: torch.nn.functional.relu
    # We apply relu to the tensor with the problematic shape (20, 0)
    var_node_0 = F.relu(arg_0)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Create a tensor with shape (20, 0) using as_strided, similar to the original bug report.
# Using float32 as relu is typically used with floating point numbers.
arg_0 = torch.as_strided(torch.randn(20).to(torch.float32), (20, 0), (1, 20))

args = (arg_0,) + (sentinel,)

# Run in eager mode
result_original = fuzzed_program(*args)
print(' eager success')

# Run in compiled mode
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')

# Verify that the results match
assert torch.equal(result_original, result_compiled), "Eager and compiled results differ"