import torch

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(1014698)

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0 # size=(20, 0), stride=(1, 20), dtype=float32, device=cuda
    # Replaced torch.add with torch.exp
    # Note: torch.exp requires floating point input, so arg_0 is adapted to float32
    var_node_0 = torch.exp(var_node_1)
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Adapted input generation: using float32 for torch.exp compatibility and .cuda() for device consistency
arg_0 = torch.as_strided(torch.randn(20).to(torch.float32).cuda(), (20, 0), (1, 20))

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')

# Verify results match
assert torch.allclose(result_original, result_compiled)