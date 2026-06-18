import torch

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(1166094474)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    var_node_3 = torch.full((12,), False, dtype=torch.bool)
    
    # Replace torch.chunk(var_node_3, 4, dim=0)[0] with torch.narrow
    # chunk splits size 12 into 4 parts of size 3. [0] takes the first one.
    # narrow takes the first 3 elements starting at index 0.
    var_node_2 = torch.narrow(var_node_3, 0, 0, 3)

    var_node_6 = arg_0
    var_node_7 = arg_1
    _input_size_var_node_5 = var_node_6.size(0)
    _index_var_node_5 = torch.randint(0, _input_size_var_node_5, (10,), device=var_node_6.device)
    var_node_5 = torch.gather(var_node_6, 0, _index_var_node_5)

    # Replace torch.chunk(var_node_5, 2, dim=0)[0] with torch.narrow
    # chunk splits size 10 into 2 parts of size 5. [0] takes the first one.
    # narrow takes the first 5 elements starting at index 0.
    var_node_4 = torch.narrow(var_node_5, 0, 0, 5)

    var_node_10 = arg_2
    
    # Replace torch.chunk(var_node_10, 4, dim=1)[0] with torch.narrow
    # chunk splits dim 1 (size 4) into 4 parts of size 1. [0] takes the first one.
    # narrow takes the first 1 element in dim 1 starting at index 0.
    var_node_9 = torch.narrow(var_node_10, 1, 0, 1)

    var_node_8 = torch.squeeze(var_node_9)
    var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0)
    var_node_11 = arg_3
    var_node_0 = torch.cat([var_node_1, var_node_11], dim=0)

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Input arguments
arg_0 = torch.as_strided(torch.randint(0, 2, (12,), dtype=torch.int8).bool(), (12,), (1,))
arg_1 = torch.as_strided(torch.randint(5, 30, (10,)).to(torch.int64), (10,), (1,))
arg_2 = torch.as_strided(torch.randint(0, 2, (24,), dtype=torch.int8).bool(), (6, 4), (4, 1))
arg_3 = torch.as_strided(torch.randint(0, 2, (2,), dtype=torch.int8).bool(), (2,), (1,))

args = (arg_0, arg_1, arg_2, arg_3) + (sentinel,)

# Run Eager
print("Running eager execution...")
result_original = fuzzed_program(*args)
print(' eager success')

# Run Compiled
print("Running compiled execution...")
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')

# Verify results match
assert torch.allclose(result_original, result_compiled), "Eager and compiled results differ"
print(" Test passed: Results match.")