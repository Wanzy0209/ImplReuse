import torch
import torch._dynamo

# Reproduce the configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(1014698)

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0 # size=(20, 0), stride=(1, 20), dtype=int64, device=cuda
    
    # Setup logic to create the specific tensor context
    var_node_3 = torch.full((), True, dtype=torch.bool) # size=(), stride=(), dtype=bool, device=cuda
    _x_nz = torch.zeros((), dtype=torch.bool, device=var_node_3.device)
    _x_nz_flat = _x_nz.reshape(-1)
    _x_nz_flat[:20] = True
    var_node_2 = torch.nonzero(_x_nz) # size=(20, 0), stride=(1, 20), dtype=int64, device=cuda
    
    # Adaptation: Replace torch.add with torch.numel
    # We test numel on the tensor with the specific (20, 0) shape
    var_node_0 = torch.numel(var_node_1)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Create the specific strided tensor argument
arg_0 = torch.as_strided(torch.randint(5, 30, (20,)).to(torch.int64), (20, 0), (1, 20))

args = (arg_0,) + (sentinel,)

# Run in eager mode
result_original = fuzzed_program(*args)
print(' eager success')

# Run in compiled mode
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')

# Verify that the results match to catch eager/compile divergence
assert torch.equal(result_original, result_compiled), \
    f"Divergence detected: Eager result {result_original} != Compiled result {result_compiled}"