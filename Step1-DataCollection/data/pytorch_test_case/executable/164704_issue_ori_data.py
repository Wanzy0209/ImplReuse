import torch
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(114503)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_4 = torch.full((2, 3), 3, dtype=torch.int16) # size=(2, 3), stride=(3, 1), dtype=int16, device=cuda
    var_node_3 = torch.unique(var_node_4) # size=(1,), stride=(1,), dtype=int16, device=cuda
    var_node_2 = torch.squeeze(var_node_3) # size=(), stride=(), dtype=int16, device=cuda
    var_node_7 = arg_0 # size=(), stride=(), dtype=int16, device=cuda
    var_node_8 = arg_1 # size=(), stride=(), dtype=int16, device=cuda
    var_node_6 = torch.sub(var_node_7, var_node_8) # size=(), stride=(), dtype=int16, device=cuda
    var_node_10 = torch.full((1,), 3, dtype=torch.int16) # size=(1,), stride=(1,), dtype=int16, device=cuda
    var_node_9 = torch.squeeze(var_node_10) # size=(), stride=(), dtype=int16, device=cuda
    var_node_5 = torch.add(var_node_6, var_node_9) # size=(), stride=(), dtype=int16, device=cuda
    var_node_1 = torch.div(var_node_2, var_node_5) # size=(), stride=(), dtype=int16, device=cuda
    var_node_0 = var_node_1.item() # dtype=int16
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.as_strided(torch.randint(5, 30, (1,)).to(torch.int16), (), ())
arg_1 = torch.as_strided(torch.randint(5, 30, (1,)).to(torch.int16), (), ())

args = (arg_0, arg_1) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print('✅ compile success')