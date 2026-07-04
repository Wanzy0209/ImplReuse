import torch
from torch.distributions.constraints import _Cat, boolean

# Fix: Check if torch._dynamo exists before accessing its config
# This handles environments where PyTorch < 2.0 is used or the attribute is not exposed
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1061983224)

def fuzzed_program(arg_0, sentinel):
    var_node_4 = arg_0 # size=(4,), stride=(1,), dtype=bool, device=cuda
    var_node_3 = torch.chunk(var_node_4, 4, dim=0)[0] # size=(1,), stride=(1,), dtype=bool, device=cuda
    var_node_2 = torch.squeeze(var_node_3) # size=(), stride=(), dtype=bool, device=cuda
    
    # Adaptation: Use torch.distributions.constraints.cat
    # We instantiate the _Cat constraint and call check on the tensor.
    # Note: _Cat is a constraint class, so we call its check method rather than 
    # replacing the tensor operation directly.
    cseq = [boolean(), boolean()]
    cat_constraint = _Cat(cseq, dim=0)
    cat_constraint.check(var_node_4)
    
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

# Fix: Check if torch.compile is available to prevent AttributeError in older PyTorch versions
if hasattr(torch, 'compile'):
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
else:
    print(' compile skipped (torch.compile not available)')