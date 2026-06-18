import torch

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(1012969)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0 # size=(2, 10), stride=(10, 1), dtype=float64, device=cuda
    var_node_4 = arg_1 # size=(10, 3), stride=(3, 1), dtype=float64, device=cuda
    var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_4.to(torch.float64)) # size=(2, 3), stride=(3, 1), dtype=float64, device=cuda
    
    # Adaptation: Replace torch.unique with torch.fliplr
    # torch.fliplr requires a 2D input. We reshape the arange output to (1, 1)
    # to satisfy the 2D requirement while keeping the matmul logic valid
    # (1, 1) @ (1, 18) -> (1, 18).
    _inp_fliplr = torch.arange(1, device=var_node_2.device, dtype=torch.int64).view(1, 1)
    _flipped = torch.fliplr(_inp_fliplr)
    var_node_1 = _flipped.to(var_node_2.dtype) # size=(1, 1), stride=(1, 1), dtype=float64, device=cuda
    
    var_node_5 = torch.full((1, 18), 0.40330381448978797, dtype=torch.float64) # size=(1, 18), stride=(18, 1), dtype=float64, device=cuda
    var_node_0 = torch.matmul(var_node_1.to(torch.float64), var_node_5.to(torch.float64)) # size=(1, 18), stride=(18, 1), dtype=float64, device=cuda
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.as_strided(torch.randn(20).to(torch.float64), (2, 10), (10, 1))
arg_1 = torch.as_strided(torch.randn(30).to(torch.float64), (10, 3), (3, 1))

args = (arg_0, arg_1) + (sentinel,)

# Test Eager Mode
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Test Compiled Mode
try:
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')