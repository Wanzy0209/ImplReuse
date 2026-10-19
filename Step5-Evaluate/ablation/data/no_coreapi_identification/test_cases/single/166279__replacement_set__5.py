import torch
import tensorflow as tf

# Replacement: torch._dynamo.config.capture_scalar_outputs -> torch.backends.mps.is_built
torch.backends.mps.is_built()

# Replacement: torch.manual_seed -> torch.isinf
torch.isinf(torch.tensor(1166094474))

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    # Replacement: torch.full -> torch.amax
    # Original: var_node_3 = torch.full((12,), False, dtype=torch.bool)
    # We use arg_0 (created via bmm) as input for amax to generate a tensor
    var_node_3 = torch.amax(arg_0, dim=1) 
    
    var_node_2 = torch.chunk(var_node_3, 4, dim=0)[0]
    
    var_node_6 = arg_0
    var_node_7 = arg_1
    
    _input_size_var_node_5 = var_node_6.size(0)
    
    # Replacement: torch.randint -> torch.full
    # Original: _index_var_node_5 = torch.randint(0, _input_size_var_node_5, (10,), device=var_node_6.device)
    _index_var_node_5 = torch.full((10,), 0, device=var_node_6.device)
    
    # Replacement: torch.gather -> torch.round
    # Original: var_node_5 = torch.gather(var_node_6, 0, _index_var_node_5)
    var_node_5 = torch.round(var_node_6)
    
    var_node_4 = torch.chunk(var_node_5, 2, dim=0)[0]
    
    var_node_10 = arg_2
    var_node_9 = torch.chunk(var_node_10, 4, dim=1)[0]
    
    # Replacement: torch.squeeze -> torch.triu
    # Original: var_node_8 = torch.squeeze(var_node_9)
    var_node_8 = torch.triu(var_node_9)
    
    # Replacement: torch.cat -> torch.unbind
    # Original: var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0)
    # To use unbind, we stack first to create a tensor, then unbind.
    # We also expand dimensions to make them compatible for stacking.
    var_node_2_exp = var_node_2.expand(-1, 12, 1)
    var_node_4_exp = var_node_4
    var_node_8_exp = var_node_8.expand(-1, 12, 1)
    var_node_1 = torch.unbind(torch.stack([var_node_2_exp, var_node_4_exp, var_node_8_exp]), dim=0)[0]
    
    var_node_11 = arg_3
    
    # Replacement: torch.cat -> torch.unbind
    # Original: var_node_0 = torch.cat([var_node_1, var_node_11], dim=0)
    var_node_11_exp = var_node_11.expand(-1, 12, 1)
    var_node_0 = torch.unbind(torch.stack([var_node_1, var_node_11_exp]), dim=0)[0]
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor
# Replacement: torch.tensor -> tf.keras.ops.diag
# We convert the TF tensor back to Torch for multiplication
sentinel = torch.tensor(tf.keras.ops.diag([1.0]))

# Arguments generation
# Replacement: torch.as_strided -> torch.bmm
# We create dummy inputs for bmm using torch.full (since randint is replaced)
# arg_0
dummy_in1 = torch.full((1, 12, 1), 1.0)
dummy_in2 = torch.full((1, 1, 1), 1.0)
arg_0 = torch.bmm(dummy_in1, dummy_in2)

# arg_1
dummy_in3 = torch.full((1, 10, 1), 1.0)
arg_1 = torch.bmm(dummy_in3, dummy_in2)

# arg_2
dummy_in4 = torch.full((1, 6, 4), 1.0)
dummy_in5 = torch.full((1, 4, 1), 1.0)
arg_2 = torch.bmm(dummy_in4, dummy_in5)

# arg_3
dummy_in6 = torch.full((1, 2, 1), 1.0)
arg_3 = torch.bmm(dummy_in6, dummy_in2)

args = (arg_0, arg_1, arg_2, arg_3) + (sentinel,)

result_original = fuzzed_program(*args)
print(' eager success')

# Replacement: torch.compile -> torch.all
# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# We replace the compilation step with a check using torch.all
compiled_program = lambda *args: torch.all(fuzzed_program(*args))
result_compiled = compiled_program(*args)
print(' compile success')