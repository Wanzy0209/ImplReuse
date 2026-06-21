import torch

# Configuration from the original bug report
# Guard against older PyTorch versions where _dynamo does not exist
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(19990)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = arg_0 # size=(1,), stride=(1,), dtype=bool, device=cuda
    var_node_1 = torch.squeeze(var_node_2) # size=(), stride=(), dtype=bool, device=cuda
    var_node_0 = var_node_1.item() # dtype=bool
    
    # Adapted call site: Replaced multiplication (var_node_0 * sentinel) 
    # with the similar API torch.tanh applied to the tensor sentinel.
    result = torch.tanh(sentinel)
    
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.randint(0, 2, (1,), dtype=torch.bool) > 0

args = (arg_0,) + (sentinel,)

# Test in eager mode
result_original = fuzzed_program(*args)
print(' eager success')

# Test in compiled mode
# Check if torch.compile is available
if hasattr(torch, 'compile'):
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
else:
    print(' compile skipped (torch.compile not available)')