import torch

# Fix: Handle cases where torch._dynamo might not be available in the environment
try:
    torch._dynamo.config.capture_scalar_outputs = True
except AttributeError:
    pass

torch.manual_seed(974450504)

def fuzzed_program(arg_0, arg_1, sentinel):
    # Adapted to float for pdist (matrix multiplication support)
    var_node_3 = arg_0 # size=(17, 30, 17, 3), stride=(1530, 51, 3, 1), dtype=float, device=cuda
    var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0] # size=(17, 30, 17, 1), stride=(510, 17, 1, 1), dtype=float, device=cuda
    var_node_5 = torch.full((17,), 3, dtype=torch.int64) # size=(17,), stride=(1,), dtype=int64, device=cuda
    var_node_6 = arg_1 # size=(15,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_4 = var_node_5.size(0)
    _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4) # size=(15,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_1 = var_node_2.size(0)
    _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1) # size=(15, 30, 17, 1), stride=(510, 17, 1, 1), dtype=float, device=cuda
    
    # Adaptation: pdist requires 2D input. Reshape (15, 30, 17, 1) -> (15, 510)
    var_node_1_2d = var_node_1.reshape(15, -1)
    
    # Original API: torch.squeeze
    # Similar API: torch.nn.functional.pdist
    var_node_0 = torch.nn.functional.pdist(var_node_1_2d, p=2)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Inputs adapted to float and CUDA to match original context and pdist requirements
arg_0 = torch.as_strided(torch.randn(26010), (17, 30, 17, 3), (1530, 51, 3, 1)).cuda()
arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,)).cuda()

args = (arg_0, arg_1) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')