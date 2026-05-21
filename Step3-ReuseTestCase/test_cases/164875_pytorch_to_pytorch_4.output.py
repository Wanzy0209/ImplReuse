import torch

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(1014698)

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0 # size=(20, 0), stride=(1, 20), dtype=int64
    
    # Adaptation: Replace torch.add(var_node_1, var_node_2) with torch.square(var_node_1)
    # Note: var_node_2 generation is removed as torch.square is a unary operation.
    var_node_0 = torch.square(var_node_1)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Input tensor with shape (20, 0)
arg_0 = torch.as_strided(torch.randint(5, 30, (20,)).to(torch.int64), (20, 0), (1, 20))

args = (arg_0, sentinel)

# Test Eager mode
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')
    raise

# Test Compiled mode
try:
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
    
    # Verify consistency
    assert torch.equal(result_original, result_compiled), "Divergence detected between eager and compiled results"
    print(' results match')
except Exception as e:
    print(f' compile failed: {e}')
    raise