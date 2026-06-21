import torch

# Configuration from the original bug report
# Fix: Check if _dynamo exists before accessing it to handle older PyTorch versions
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1166094474)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    # Original: var_node_3 = torch.full((12,), False, dtype=torch.bool) # size=(12,), stride=(97,)
    # To safely reproduce the stride=(97,) behavior without OOM errors, 
    # we create a larger base tensor and apply the stride.
    base = torch.full((1100,), False, dtype=torch.bool)
    var_node_3 = torch.as_strided(base, (12,), (97,))

    # Original: var_node_2 = torch.chunk(var_node_3, 4, dim=0)[0]
    # Adaptation: torch.chunk splits dim 0 (size 12) into 4 chunks of size 3. [0] takes the first.
    # We use torch.as_strided to view the first 3 elements with the original stride.
    var_node_2 = torch.as_strided(var_node_3, (3,), (97,), storage_offset=0)

    var_node_6 = arg_0
    var_node_7 = arg_1
    _input_size_var_node_5 = var_node_6.size(0)
    _index_var_node_5 = torch.randint(0, _input_size_var_node_5, (10,), device=var_node_6.device)
    var_node_5 = torch.gather(var_node_6, 0, _index_var_node_5)

    # Original: var_node_4 = torch.chunk(var_node_5, 2, dim=0)[0]
    # Adaptation: torch.chunk splits dim 0 (size 10) into 2 chunks of size 5. [0] takes the first.
    var_node_4 = torch.as_strided(var_node_5, (5,), (1,), storage_offset=0)

    var_node_10 = arg_2

    # Original: var_node_9 = torch.chunk(var_node_10, 4, dim=1)[0]
    # Adaptation: torch.chunk splits dim 1 (size 4) into 4 chunks of size 1. [0] takes the first.
    # Shape (6, 4) -> (6, 1). Stride (4, 1) -> (4, 1).
    var_node_9 = torch.as_strided(var_node_10, (6, 1), (4, 1), storage_offset=0)

    var_node_8 = torch.squeeze(var_node_9)
    var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0)
    var_node_11 = arg_3
    var_node_0 = torch.cat([var_node_1, var_node_11], dim=0)

    # Ensure gradient computation
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor
sentinel = torch.tensor(1.0, requires_grad=True)

# Input generation using torch.as_strided (as per the original test case)
arg_0 = torch.as_strided(torch.randint(0, 2, (12,), dtype=torch.int8).bool(), (12,), (1,))
arg_1 = torch.as_strided(torch.randint(5, 30, (10,)).to(torch.int64), (10,), (1,))
arg_2 = torch.as_strided(torch.randint(0, 2, (24,), dtype=torch.int8).bool(), (6, 4), (4, 1))
arg_3 = torch.as_strided(torch.randint(0, 2, (2,), dtype=torch.int8).bool(), (2,), (1,))

args = (arg_0, arg_1, arg_2, arg_3) + (sentinel,)

# Run Eager
result_original = fuzzed_program(*args)
print(' eager success')

# Run Compiled
# Fix: Check if torch.compile is available to handle missing dependencies
if hasattr(torch, 'compile'):
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')

    # Verify results match
    assert torch.equal(result_original, result_compiled), "Eager and compiled results differ"
else:
    print("Warning: torch.compile is not available in this environment. Skipping compiled execution and assertion.")