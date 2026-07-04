import torch

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(974450504)

def fuzzed_program(arg_0, arg_1, sentinel):
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
    
    # --- ADAPTATION ---
    # Original API: torch.squeeze
    # Similar API: torch.zeros_like
    # Replacing torch.squeeze(var_node_1) with torch.zeros_like(var_node_1)
    var_node_0 = torch.zeros_like(var_node_1)
    # ------------------

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Input generation
# Note: The bug report comments indicate device=cuda. 
# We attempt to use CUDA if available to match the bug context, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arg_0 = torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), (17, 30, 17, 3), (1530, 51, 3, 1)).to(device)
arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,)).to(device)
sentinel = sentinel.to(device)

args = (arg_0, arg_1, sentinel)

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