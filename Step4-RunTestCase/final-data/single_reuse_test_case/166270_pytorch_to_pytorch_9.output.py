import torch

# Fix: Check if torch._dynamo exists to prevent AttributeError in older PyTorch versions
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    can_compile = True
else:
    can_compile = False
    print("Warning: torch._dynamo is not available. Skipping compilation steps.")

torch.manual_seed(1061983224)

def fuzzed_program(arg_0, sentinel):
    var_node_4 = arg_0 # size=(4,), stride=(1,), dtype=bool, device=cuda
    var_node_3 = torch.chunk(var_node_4, 4, dim=0)[0] # size=(1,), stride=(1,), dtype=bool, device=cuda
    
    # Replaced torch.squeeze with torch.empty_strided to test the similar API
    # torch.squeeze(var_node_3) results in a 0-d tensor (size=(), stride=())
    # We mimic this by creating a 0-d tensor directly with empty_strided
    var_node_2 = torch.empty_strided((), (), dtype=var_node_3.dtype, device=var_node_3.device)
    
    var_node_1 = torch.stack([var_node_2], dim=0) # size=(1,), stride=(1,), dtype=bool, device=cuda
    var_node_0 = torch.reshape(var_node_1, [1]) # size=(1,), stride=(1,), dtype=bool, device=cuda
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.as_strided(torch.randint(0, 2, (4,), dtype=torch.int8).bool(), (4,), (1,))

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

if can_compile and hasattr(torch, 'compile'):
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
else:
    print(' compile skipped (torch._dynamo or torch.compile not available)')