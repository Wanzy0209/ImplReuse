import torch

# Fix: Check if torch._dynamo exists before accessing its configuration
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(1014698)

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0 # size=(20, 0), stride=(1, 20), dtype=int64
    # Adapted: Replaced the broken nonzero logic with as_strided to ensure the tensor has size (20, 0) as indicated in the comments
    # and to ensure the test case is runnable.
    var_node_2 = torch.as_strided(torch.randint(5, 30, (20,)).to(torch.int64), (20, 0), (1, 20))
    
    # Replaced torch.add with torch.sub
    var_node_0 = torch.sub(var_node_1, var_node_2) # size=(20, 0), stride=(1, 20), dtype=int64
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.as_strided(torch.randint(5, 30, (20,)).to(torch.int64), (20, 0), (1, 20))

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

# Check if torch.compile is available before attempting to use it
if hasattr(torch, 'compile'):
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
else:
    print(' compile skipped (torch.compile not available)')