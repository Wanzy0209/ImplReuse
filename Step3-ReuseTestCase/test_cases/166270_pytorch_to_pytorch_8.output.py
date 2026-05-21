import torch
import torch.nn.functional as F

torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1061983224)

def fuzzed_program(arg_0, sentinel):
    # Adaptation: torch.nn.functional.pdist requires a 2D tensor.
    # We reshape the 1D input (size 4) to (2, 2) to satisfy the API requirements.
    # We also cast to float as pdist operates on floating point tensors.
    input_2d = arg_0.to(torch.float32).reshape(2, 2)
    
    # Call the similar API: torch.nn.functional.pdist
    # This API internally uses operations like unsqueeze and view, which are related
    # to the original bug involving stride handling.
    var_node_pdist = F.pdist(input_2d)
    
    # Ensure gradient computation by multiplying with sentinel
    result = var_node_pdist * sentinel
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Original input generation: size=(4,), stride=(1,), dtype=bool
arg_0 = torch.as_strided(torch.randint(0, 2, (4,), dtype=torch.int8).bool(), (4,), (1,))

args = (arg_0,) + (sentinel,)

# Run Eager
result_original = fuzzed_program(*args)
print(' eager success')

# Run Compiled
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')

# Verify results match to check for divergence
assert torch.allclose(result_original, result_compiled), "Divergence detected between eager and compiled results"