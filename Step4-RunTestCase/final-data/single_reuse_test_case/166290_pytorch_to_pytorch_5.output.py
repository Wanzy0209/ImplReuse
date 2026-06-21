import torch

# Configuration from the original bug report
# Fix: Check if _dynamo exists before accessing it to handle older PyTorch versions
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
else:
    print("Warning: torch._dynamo not found. Skipping dynamo config.")

torch.manual_seed(974450504)

def fuzzed_program(arg_0, arg_1, sentinel):
    # Setup logic identical to original to preserve tensor properties
    var_node_3 = arg_0 # size=(17, 30, 17, 3), stride=(1530, 51, 3, 1), dtype=bool, device=cuda
    var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0] # size=(17, 30, 17, 1), stride=(510, 17, 1, 1), dtype=bool, device=cuda
    var_node_5 = torch.full((17,), 3, dtype=torch.int64) # size=(17,), stride=(1,), dtype=int64, device=cuda
    var_node_6 = arg_1 # size=(15,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_4 = var_node_5.size(0)
    _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4) # size=(15,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_1 = var_node_2.size(0)
    _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1) # size=(15, 30, 17, 1), stride=(510, 17, 1, 1), dtype=bool, device=cuda
    
    # REPLACEMENT: torch.squeeze -> torch.triu
    # var_node_1 is (15, 30, 17, 1). torch.triu will operate on the last two dimensions (17, 1).
    var_node_0 = torch.triu(var_node_1)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Input generation (identical to original to maintain specific strides/dtypes)
arg_0 = torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), (17, 30, 17, 3), (1530, 51, 3, 1))
arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,))

args = (arg_0, arg_1) + (sentinel,)

# Test Eager Execution
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Test Compiled Execution
# Fix: Check if torch.compile exists
if hasattr(torch, 'compile'):
    try:
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        result_compiled = compiled_program(*args)
        print(' compile success')
    except Exception as e:
        print(f' compile failed: {e}')
else:
    print(' compile skipped: torch.compile not available')