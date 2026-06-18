import torch
torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1215252001)

def fuzzed_program(arg_0, arg_1, arg_2, sentinel):
    var_node_4 = arg_0 # size=(9, 1, 15, 4), stride=(60, 60, 0, 1), dtype=int32, device=cuda
    var_node_3 = torch.squeeze(var_node_4) # size=(9, 15, 4), stride=(60, 4, 1), dtype=int32, device=cuda
    var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0] # size=(9, 15, 1), stride=(1, 1, 1), dtype=int32, device=cuda
    var_node_1 = torch.squeeze(var_node_2) # size=(9, 15), stride=(0, 1), dtype=int32, device=cuda
    var_node_7 = arg_1 # size=(20, 15), stride=(15, 1), dtype=int64, device=cuda
    var_node_8 = arg_2 # size=(18, 15), stride=(15, 1), dtype=int64, device=cuda
    _input_size_var_node_6 = var_node_7.size(0)
    _index_var_node_6 = torch.randint(0, _input_size_var_node_6, (18, 15), device=var_node_7.device)
    var_node_6 = torch.gather(var_node_7, 0, _index_var_node_6) # size=(18, 15), stride=(0, 0), dtype=int64, device=cuda
    var_node_5 = torch.chunk(var_node_6, 2, dim=0)[0] # size=(9, 15), stride=(0, 1), dtype=int64, device=cuda
    var_node_0 = torch.mul(var_node_1, var_node_5) # size=(9, 15), stride=(0, 1), dtype=int64, device=cuda
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.as_strided(torch.randint(5, 30, (484,)).to(torch.int32), (9, 1, 15, 4), (60, 60, 0, 1))
arg_1 = torch.as_strided(torch.randint(5, 30, (300,)).to(torch.int64), (20, 15), (15, 1))
arg_2 = torch.as_strided(torch.randint(5, 30, (270,)).to(torch.int64), (18, 15), (15, 1))

args = (arg_0, arg_1, arg_2) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print('✅ compile success')