import torch

torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(1166094474)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    # Adapted to use float dtype for compatibility with bucketize and other ops
    var_node_3 = torch.full((12,), 0.5, dtype=torch.float) # size=(12,)

    # Original: var_node_2 = torch.chunk(var_node_3, 4, dim=0)[0]
    # Adapted: Replace torch.chunk with torch.bucketize
    # arg_1 is used as the boundaries tensor. 
    # We cast the result to float to ensure compatibility with subsequent torch.cat operations.
    var_node_2 = torch.bucketize(var_node_3, arg_1).to(torch.float) # size=(12,)

    var_node_6 = arg_0 # size=(12,)
    _input_size_var_node_5 = var_node_6.size(0)
    _index_var_node_5 = torch.randint(0, _input_size_var_node_5, (10,), device=var_node_6.device)
    var_node_5 = torch.gather(var_node_6, 0, _index_var_node_5) # size=(10,)

    # Original: var_node_4 = torch.chunk(var_node_5, 2, dim=0)[0]
    # Adapted: Replace torch.chunk with torch.bucketize
    var_node_4 = torch.bucketize(var_node_5, arg_1).to(torch.float) # size=(10,)

    var_node_10 = arg_2 # size=(6, 4)
    var_node_9 = torch.chunk(var_node_10, 4, dim=1)[0] # size=(6, 1)
    var_node_8 = torch.squeeze(var_node_9) # size=(6,)

    var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0) # size=(28,)
    var_node_11 = arg_3 # size=(2,)
    var_node_0 = torch.cat([var_node_1, var_node_11], dim=0) # size=(30,)

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Inputs adapted for torch.bucketize
# arg_0: Input tensor for gather/bucketize
arg_0 = torch.as_strided(torch.randn(12), (12,), (1,))
# arg_1: Boundaries for bucketize (must be 1D)
arg_1 = torch.as_strided(torch.sort(torch.randint(0, 10, (10,))).values, (10,), (1,))
# arg_2: Input tensor for chunk/squeeze
arg_2 = torch.as_strided(torch.randn(24), (6, 4), (4, 1))
# arg_3: Input tensor for final cat
arg_3 = torch.as_strided(torch.randn(2), (2,), (1,))

args = (arg_0, arg_1, arg_2, arg_3) + (sentinel,)

# Test Eager mode
result_original = fuzzed_program(*args)
print(' eager success')

# Test Compiled mode
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')