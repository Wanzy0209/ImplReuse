import torch
import sys

# Fix: Check if torch._dynamo is available before accessing its config
# This handles environments where PyTorch version is < 2.0 or _dynamo is not exposed
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
else:
    print("torch._dynamo is not available in this environment. Skipping test.")
    sys.exit(0)

torch.manual_seed(19989)

def fuzzed_program(arg_0, sentinel):
    # Adapted to use torch.all instead of scalar division
    # arg_0 is a tensor
    var_node_0 = torch.all(arg_0)  # dtype=bool (scalar)
    
    # Ensure gradient computation by multiplying with sentinel
    # Cast to float to multiply with sentinel (float)
    result = var_node_0.float() * sentinel
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# arg_0 is now a tensor, not a scalar item, to be used with torch.all
arg_0 = torch.tensor(torch.randn(()), dtype=torch.int32)

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')

# Verify results match
assert torch.allclose(result_original, result_compiled)