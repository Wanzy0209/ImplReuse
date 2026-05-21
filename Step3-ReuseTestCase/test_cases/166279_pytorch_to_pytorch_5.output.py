import torch
import torch.nn.functional as F

torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1166094474)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    # Adaptation: Replace the original torch.chunk/cat logic with torch.nn.functional.pdist.
    # pdist requires a 2D float tensor. We use arg_2 which is 2D.
    # We cast to float to satisfy pdist requirements.
    input_tensor = arg_2.to(torch.float32)
    
    # Call the similar API: torch.nn.functional.pdist
    # This computes the p-norm distance between each pair of row vectors.
    var_node_pdist = F.pdist(input_tensor, p=2)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_pdist * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Setup arguments
# arg_0, arg_1, arg_3 are kept to match the original signature, though unused in the adapted logic
arg_0 = torch.as_strided(torch.randint(0, 2, (12,), dtype=torch.int8).bool(), (12,), (1,))
arg_1 = torch.as_strided(torch.randint(5, 30, (10,)).to(torch.int64), (10,), (1,))
# arg_2 is modified to be a float tensor suitable for pdist, maintaining the original shape (6, 4)
arg_2 = torch.as_strided(torch.randn(24), (6, 4), (4, 1))
arg_3 = torch.as_strided(torch.randint(0, 2, (2,), dtype=torch.int8).bool(), (2,), (1,))

args = (arg_0, arg_1, arg_2, arg_3, sentinel)

# Run in eager mode
result_original = fuzzed_program(*args)
print(' eager success')

# Run in compiled mode
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')

# Verify results match
assert torch.allclose(result_original, result_compiled), "Divergence between eager and compiled modes"