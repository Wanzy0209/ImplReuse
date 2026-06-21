import torch

# Fix: Check if torch._dynamo exists before accessing its attributes
# This handles older PyTorch versions or environments where dynamo is not exposed
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(19990)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = arg_0 # size=(1,), stride=(1,), dtype=bool, device=cuda
    var_node_1 = torch.squeeze(var_node_2) # size=(), stride=(), dtype=bool, device=cuda
    var_node_0 = var_node_1.item() # dtype=bool
    # Adaptation: Apply torch.exp to the scalar and multiply by sentinel to ensure gradient computation
    result = torch.exp(var_node_0) * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.randint(0, 2, (1,), dtype=torch.bool) > 0

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

# Fix: Check if torch.compile is available before attempting to use it
if hasattr(torch, 'compile'):
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
else:
    print(' compile skipped (torch.compile not available)')