import torch

# Replacements applied based on the provided mapping:
# torch.manual_seed -> torch.argmax
# torch.as_strided -> torch.logcumsumexp
# torch.squeeze -> torch.cat (Interpreted as torch.cat for runnability)
# torch.chunk -> torch.empty_strided
# torch.gather -> torch.trunc
# torch.mul -> torch.log
# torch.randint -> torch.argmin
# torch.tensor -> torch.cummax

torch._dynamo.config.capture_scalar_outputs = True

# torch.manual_seed(1215252001) -> torch.argmax(...)
torch.argmax(torch.tensor([1215252001]))

def fuzzed_program(arg_0, arg_1, arg_2, sentinel):
    var_node_4 = arg_0
    
    # var_node_3 = torch.squeeze(var_node_4) -> torch.cat(...)
    # Using torch.cat to maintain tensor flow, as constraints.cat is not a tensor op
    var_node_3 = torch.cat([var_node_4], dim=0)
    
    # var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0] -> torch.empty_strided(...)
    # Original size=(9, 15, 1), stride=(1, 1, 1)
    var_node_2 = torch.empty_strided((9, 15, 1), (1, 1, 1), device=var_node_3.device, dtype=var_node_3.dtype)
    
    # var_node_1 = torch.squeeze(var_node_2) -> torch.cat(...)
    var_node_1 = torch.cat([var_node_2], dim=0)
    
    var_node_7 = arg_1
    var_node_8 = arg_2
    _input_size_var_node_6 = var_node_7.size(0)
    
    # _index_var_node_6 = torch.randint(0, _input_size_var_node_6, (18, 15), device=var_node_7.device)
    # -> torch.argmin(...)
    _index_var_node_6 = torch.argmin(var_node_7, dim=0)
    
    # var_node_6 = torch.gather(var_node_7, 0, _index_var_node_6) -> torch.trunc(...)
    var_node_6 = torch.trunc(var_node_7)
    
    # var_node_5 = torch.chunk(var_node_6, 2, dim=0)[0] -> torch.empty_strided(...)
    # Original size=(9, 15), stride=(0, 1)
    var_node_5 = torch.empty_strided((9, 15), (0, 1), device=var_node_6.device, dtype=var_node_6.dtype)
    
    # var_node_0 = torch.mul(var_node_1, var_node_5) -> torch.log(...)
    var_node_0 = torch.log(var_node_1)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# sentinel = torch.tensor(1.0, requires_grad=True) -> torch.cummax(...)
# cummax returns (values, indices), we take values
sentinel = torch.cummax(torch.tensor([1.0]), dim=0).values
sentinel.requires_grad_(True)

# arg_0 = torch.as_strided(...) -> torch.logcumsumexp(...)
base_0 = torch.randint(5, 30, (484,)).to(torch.int32)
arg_0 = torch.logcumsumexp(base_0, dim=0)

# arg_1 = torch.as_strided(...) -> torch.logcumsumexp(...)
base_1 = torch.randint(5, 30, (300,)).to(torch.int64)
arg_1 = torch.logcumsumexp(base_1, dim=0)

# arg_2 = torch.as_strided(...) -> torch.logcumsumexp(...)
base_2 = torch.randint(5, 30, (270,)).to(torch.int64)
arg_2 = torch.logcumsumexp(base_2, dim=0)

args = (arg_0, arg_1, arg_2) + (sentinel,)

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