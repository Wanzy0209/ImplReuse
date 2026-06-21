import torch

# Fix: Check if torch._dynamo exists before accessing it to handle older PyTorch versions
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1061983224)

def fuzzed_program(arg_0, sentinel):
    # Adapted to 2D tensor to accommodate torch.triu
    # var_node_4: size=(2, 2), stride=(1, 2), dtype=bool
    var_node_4 = arg_0
    
    # Chunk along dim 0
    # var_node_3: size=(1, 2), stride=(1, 2)
    var_node_3 = torch.chunk(var_node_4, 2, dim=0)[0]
    
    # Replace torch.squeeze with torch.triu
    # var_node_2: size=(1, 2)
    var_node_2 = torch.triu(var_node_3)
    
    # Stack and reshape to stress the compiler's view handling
    # var_node_1: size=(1, 1, 2)
    var_node_1 = torch.stack([var_node_2], dim=0)
    # var_node_0: size=(2,)
    var_node_0 = torch.reshape(var_node_1, [2])
    
    # Ensure gradient computation
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Adapt input to be 2D for torch.triu, maintaining non-contiguous stride
# Original was (4,), (1,). New is (2, 2), (1, 2).
base = torch.randint(0, 2, (4,), dtype=torch.int8).bool()
arg_0 = torch.as_strided(base, (2, 2), (1, 2))

args = (arg_0, sentinel)

# Test Eager mode
result_original = fuzzed_program(*args)
print(' eager success')

# Test Compiled mode
# Fix: Check if torch.compile is available to handle environments without torch.compile
if hasattr(torch, 'compile'):
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')

    # Verify consistency
    assert torch.allclose(result_original, result_compiled), "Eager and compiled results differ"
else:
    print(' compile skipped (torch.compile not available)')