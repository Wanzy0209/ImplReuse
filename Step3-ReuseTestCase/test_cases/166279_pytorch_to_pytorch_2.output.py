import torch

torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1166094474)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    var_node_3 = torch.full((12,), False, dtype=torch.bool)
    
    # Adaptation: Replace torch.chunk with torch.logcumsumexp
    # Original: torch.chunk(var_node_3, 4, dim=0)[0] -> size (3,)
    # New: logcumsumexp preserves size (12,), slice to (3,) to maintain compatibility
    var_node_2 = torch.logcumsumexp(var_node_3.to(torch.float32), dim=0)[:3]
    
    var_node_6 = arg_0
    var_node_7 = arg_1
    _input_size_var_node_5 = var_node_6.size(0)
    _index_var_node_5 = torch.randint(0, _input_size_var_node_5, (10,), device=var_node_6.device)
    var_node_5 = torch.gather(var_node_6, 0, _index_var_node_5)
    
    # Adaptation: Replace torch.chunk with torch.logcumsumexp
    # Original: torch.chunk(var_node_5, 2, dim=0)[0] -> size (5,)
    # New: logcumsumexp preserves size (10,), slice to (5,)
    var_node_4 = torch.logcumsumexp(var_node_5.to(torch.float32), dim=0)[:5]
    
    var_node_10 = arg_2
    # Adaptation: Replace torch.chunk with torch.logcumsumexp
    # Original: torch.chunk(var_node_10, 4, dim=1)[0] -> size (6, 1)
    # New: logcumsumexp preserves size (6, 4), slice to (6, 1)
    var_node_9 = torch.logcumsumexp(var_node_10.to(torch.float32), dim=1)[:, :1]
    
    var_node_8 = torch.squeeze(var_node_9)
    var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0)
    var_node_11 = arg_3
    var_node_0 = torch.cat([var_node_1, var_node_11], dim=0)
    
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

# Test Eager execution
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Test Compiled execution
try:
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')