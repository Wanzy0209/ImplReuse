import torch

# Fix: Check if _dynamo is available before accessing its config to avoid AttributeError
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1166094474)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    var_node_3 = torch.full((12,), False, dtype=torch.bool) # size=(12,), stride=(97,), dtype=bool, device=cuda
    
    # Adaptation: Replace torch.chunk(var_node_3, 4, dim=0)[0] with torch.empty
    # torch.chunk splits a tensor of size 12 into 4 chunks, resulting in size 3.
    var_node_2 = torch.empty((3,), dtype=var_node_3.dtype, device=var_node_3.device) # size=(3,), dtype=bool, device=cuda
    
    var_node_6 = arg_0 # size=(12,), stride=(1,), dtype=bool, device=cuda
    var_node_7 = arg_1 # size=(10,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_5 = var_node_6.size(0)
    _index_var_node_5 = torch.randint(0, _input_size_var_node_5, (10,), device=var_node_6.device)
    var_node_5 = torch.gather(var_node_6, 0, _index_var_node_5) # size=(10,), stride=(1,), dtype=bool, device=cuda
    
    # Adaptation: Replace torch.chunk(var_node_5, 2, dim=0)[0] with torch.empty
    # torch.chunk splits a tensor of size 10 into 2 chunks, resulting in size 5.
    var_node_4 = torch.empty((5,), dtype=var_node_5.dtype, device=var_node_5.device) # size=(5,), dtype=bool, device=cuda
    
    var_node_10 = arg_2 # size=(6, 4), stride=(4, 1), dtype=bool, device=cuda
    
    # Adaptation: Replace torch.chunk(var_node_10, 4, dim=1)[0] with torch.empty
    # torch.chunk splits dim 1 of size (6, 4) into 4 chunks, resulting in size (6, 1).
    var_node_9 = torch.empty((6, 1), dtype=var_node_10.dtype, device=var_node_10.device) # size=(6, 1), dtype=bool, device=cuda
    
    var_node_8 = torch.squeeze(var_node_9) # size=(6,), stride=(1,), dtype=bool, device=cuda
    var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0) # size=(14,), stride=(1,), dtype=bool, device=cuda
    var_node_11 = arg_3 # size=(2,), stride=(1,), dtype=bool, device=cuda
    var_node_0 = torch.cat([var_node_1, var_node_11], dim=0) # size=(16,), stride=(1,), dtype=bool, device=cuda
    
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

try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

try:
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')