import torch

# Fix: Check if _dynamo exists before accessing it to handle older PyTorch versions
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1166094474)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    var_node_3 = torch.full((12,), False, dtype=torch.bool) # size=(12,), stride=(97,), dtype=bool
    # Adapted: Replace torch.chunk with torch.triu
    # torch.triu requires at least 2D input, so we unsqueeze and squeeze
    var_node_2 = torch.triu(var_node_3.unsqueeze(0)).squeeze(0) 

    var_node_6 = arg_0 # size=(12,), stride=(1,), dtype=bool
    var_node_7 = arg_1 # size=(10,), stride=(1,), dtype=int64
    _input_size_var_node_5 = var_node_6.size(0)
    _index_var_node_5 = torch.randint(0, _input_size_var_node_5, (10,), device=var_node_6.device)
    var_node_5 = torch.gather(var_node_6, 0, _index_var_node_5) # size=(10,), stride=(1,), dtype=bool
    
    # Adapted: Replace torch.chunk with torch.triu
    # torch.triu requires at least 2D input, so we unsqueeze and squeeze
    var_node_4 = torch.triu(var_node_5.unsqueeze(0)).squeeze(0)

    var_node_10 = arg_2 # size=(6, 4), stride=(4, 1), dtype=bool
    # Adapted: Replace torch.chunk with torch.triu
    var_node_9 = torch.triu(var_node_10)

    var_node_8 = torch.squeeze(var_node_9) # size=(6,), stride=(1,), dtype=bool
    var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0) # size=(28,), stride=(1,), dtype=bool
    var_node_11 = arg_3 # size=(2,), stride=(1,), dtype=bool
    var_node_0 = torch.cat([var_node_1, var_node_11], dim=0) # size=(30,), stride=(1,), dtype=bool
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.as_strided(torch.randint(0, 2, (12,), dtype=torch.int8).bool(), (12,), (1,))
arg_1 = torch.as_strided(torch.randint(5, 30, (10,)).to(torch.int64), (10,), (1,))
arg_2 = torch.as_strided(torch.randint(0, 2, (24,), dtype=torch.int8).bool(), (6, 4), (4, 1))
arg_3 = torch.as_strided(torch.randint(0, 2, (2,), dtype=torch.int8).bool(), (2,), (1,))

args = (arg_0, arg_1, arg_2, arg_3) + (sentinel,)

# Run eager mode
result_original = fuzzed_program(*args)
print(' eager success')

# Run compiled mode
# Fix: Check if torch.compile is available
if hasattr(torch, 'compile'):
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
else:
    print(' compile skipped (torch.compile not available)')