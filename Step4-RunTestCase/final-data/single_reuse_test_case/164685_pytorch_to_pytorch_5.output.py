import torch
import sys

# Check for torch._dynamo availability
if not hasattr(torch, '_dynamo'):
    print("torch._dynamo is not available. This test requires PyTorch 2.0+.")
    sys.exit(0)

# Check for torch.compile availability
if not hasattr(torch, 'compile'):
    print("torch.compile is not available. This test requires PyTorch 2.0+.")
    sys.exit(0)

# Replicate the configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(19989)

def fuzzed_program(arg_0):
    # Adapt the logic to use torch.any
    # Original logic involved int32/int64 mixing and scalar ops
    var_node_2 = -6
    var_node_3 = arg_0.to(torch.int32)
    var_node_1 = var_node_2 * var_node_3
    
    # Use torch.any to check if any element is non-zero
    # This returns a scalar boolean tensor
    result = torch.any(var_node_1 != 0)
    
    return result

# Setup args
# Using a tensor input for torch.any
arg_0 = torch.randn(5, dtype=torch.float32)

# Run eager
result_original = fuzzed_program(arg_0)
print(' eager success')

# Compile
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(arg_0)
print(' compile success')

# Verify results match
assert result_original == result_compiled