import torch
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(13653)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = torch.full((), 1.0, dtype=torch.float32) # size=(), stride=(), dtype=float32, device=cuda
    var_node_2 = var_node_3.item() # dtype=float32
    var_node_5 = -3 # dtype=int32
    var_node_6 = arg_0 # dtype=int64
    var_node_4 = var_node_5 + var_node_6 # dtype=int32
    var_node_1 = var_node_2 + var_node_4 # dtype=float32
    var_node_9 = 1 # dtype=int64
    var_node_10 = -10 # dtype=int32
    var_node_8 = var_node_9 / var_node_10 # dtype=int64
    var_node_12 = arg_1 # dtype=int64
    var_node_13 = -5 # dtype=int32
    var_node_11 = var_node_12 / var_node_13 # dtype=int32
    var_node_7 = var_node_8 + var_node_11 # dtype=int32
    var_node_0 = var_node_1 * var_node_7 # dtype=float32
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
arg_1 = torch.tensor(torch.randn(()), dtype=torch.int64).item()

args = (arg_0, arg_1) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print('✅ compile success')