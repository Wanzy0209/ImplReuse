import torch
torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1352030645)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, sentinel):
    var_node_4 = arg_0 # size=(4, 8), stride=(8, 1), dtype=bfloat16, device=cuda
    var_node_5 = torch.full((8, 7), -0.80078125, dtype=torch.bfloat16) # size=(8, 7), stride=(7, 1), dtype=bfloat16, device=cuda
    var_node_3 = torch.matmul(var_node_4.to(torch.bfloat16), var_node_5.to(torch.bfloat16)) # size=(4, 7), stride=(7, 1), dtype=bfloat16, device=cuda
    var_node_7 = arg_1 # size=(7, 12), stride=(12, 1), dtype=bfloat16, device=cuda
    var_node_8 = arg_2 # size=(12, 2), stride=(2, 1), dtype=bfloat16, device=cuda
    var_node_6 = torch.matmul(var_node_7.to(torch.bfloat16), var_node_8.to(torch.bfloat16)) # size=(7, 2), stride=(2, 1), dtype=bfloat16, device=cuda
    var_node_2 = torch.matmul(var_node_3.to(torch.bfloat16), var_node_6.to(torch.bfloat16)) # size=(4, 2), stride=(2, 1), dtype=bfloat16, device=cuda
    var_node_11 = torch.full((2, 3), 1.515625, dtype=torch.bfloat16) # size=(2, 3), stride=(3, 1), dtype=bfloat16, device=cuda
    var_node_12 = torch.full((3, 16), 0.2353515625, dtype=torch.bfloat16) # size=(3, 16), stride=(16, 1), dtype=bfloat16, device=cuda
    var_node_10 = torch.matmul(var_node_11.to(torch.bfloat16), var_node_12.to(torch.bfloat16)) # size=(2, 16), stride=(16, 1), dtype=bfloat16, device=cuda
    var_node_14 = torch.full((16, 4), 2.21875, dtype=torch.bfloat16) # size=(16, 4), stride=(4, 1), dtype=bfloat16, device=cuda
    var_node_15 = torch.full((4, 9), -1.7421875, dtype=torch.bfloat16) # size=(4, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_13 = torch.matmul(var_node_14.to(torch.bfloat16), var_node_15.to(torch.bfloat16)) # size=(16, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_9 = torch.matmul(var_node_10.to(torch.bfloat16), var_node_13.to(torch.bfloat16)) # size=(2, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_1 = torch.matmul(var_node_2.to(torch.bfloat16), var_node_9.to(torch.bfloat16)) # size=(4, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_19 = arg_3 # size=(14, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_20 = torch.full((9, 2), 0.8203125, dtype=torch.bfloat16) # size=(9, 2), stride=(2, 1), dtype=bfloat16, device=cuda
    var_node_18 = torch.matmul(var_node_19.to(torch.bfloat16), var_node_20.to(torch.bfloat16)) # size=(14, 2), stride=(2, 1), dtype=bfloat16, device=cuda
    var_node_22 = arg_4 # size=(2,), stride=(1,), dtype=bfloat16, device=cuda
    var_node_23 = torch.full((2,), -0.7421875, dtype=torch.bfloat16) # size=(2,), stride=(1,), dtype=bfloat16, device=cuda
    var_node_21 = torch.add(var_node_22, var_node_23) # size=(2,), stride=(1,), dtype=bfloat16, device=cuda
    var_node_17 = torch.matmul(var_node_18.to(torch.bfloat16), var_node_21.to(torch.bfloat16)) # size=(14,), stride=(1,), dtype=bfloat16, device=cuda
    var_node_26 = arg_5 # size=(478, 13), stride=(13, 1), dtype=bfloat16, device=cuda
    var_node_27 = torch.full((13, 9), 0.3359375, dtype=torch.bfloat16) # size=(13, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_25 = torch.matmul(var_node_26.to(torch.bfloat16), var_node_27.to(torch.bfloat16)) # size=(478, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_29 = arg_6 # size=(14,), stride=(1,), dtype=int64, device=cuda
    var_node_30 = arg_7 # size=(14,), stride=(1,), dtype=int64, device=cuda
    var_node_28 = torch.div(var_node_29, var_node_30) # size=(14,), stride=(1,), dtype=int64, device=cuda
    var_node_24 = torch.nn.functional.embedding(torch.clamp(var_node_28.to(torch.int64), 0, var_node_25.size(0)-1), var_node_25) # size=(14, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_16 = torch.matmul(var_node_17.to(torch.bfloat16), var_node_24.to(torch.bfloat16)) # size=(9,), stride=(1,), dtype=bfloat16, device=cuda
    var_node_0 = torch.matmul(var_node_1.to(torch.bfloat16), var_node_16.to(torch.bfloat16)) # size=(4,), stride=(1,), dtype=bfloat16, device=cuda
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.as_strided(torch.randn(32).to(torch.bfloat16), (4, 8), (8, 1))
arg_1 = torch.as_strided(torch.randn(84).to(torch.bfloat16), (7, 12), (12, 1))
arg_2 = torch.as_strided(torch.randn(24).to(torch.bfloat16), (12, 2), (2, 1))
arg_3 = torch.as_strided(torch.randn(126).to(torch.bfloat16), (14, 9), (9, 1))
arg_4 = torch.as_strided(torch.randn(2).to(torch.bfloat16), (2,), (1,))
arg_5 = torch.as_strided(torch.randn(6214).to(torch.bfloat16), (478, 13), (13, 1))
arg_6 = torch.as_strided(torch.randint(5, 30, (14,)).to(torch.int64), (14,), (1,))
arg_7 = torch.as_strided(torch.randint(5, 30, (14,)).to(torch.int64), (14,), (1,))

args = (arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print('✅ compile success')