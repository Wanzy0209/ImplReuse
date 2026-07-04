import torch
import sys

# Check if torch._dynamo is available (requires PyTorch 2.0+)
if not hasattr(torch, '_dynamo'):
    print("Skipping test: torch._dynamo not found (requires PyTorch 2.0+)")
    sys.exit(0)

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(238)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = torch.full((), 1, dtype=torch.int16)
    var_node_3 = arg_0
    var_node_1 = torch.add(var_node_2, var_node_3)

    # Replaced torch.div with torch.tanh
    # Original: var_node_0 = torch.div(var_node_1, var_node_4)
    var_node_0 = torch.tanh(var_node_1)

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Input tensor
arg_0 = torch.as_strided(torch.randn(1).to(torch.int16), (), ())

args = (arg_0,) + (sentinel,)

# Eager execution
out_eager = fuzzed_program(*args)
out_eager.sum().backward()
print('Eager Success! ')

# Compiled execution
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
out_compiled = compiled_program(*args)
out_compiled.sum().backward()
print('Compile Success! ')

# Comparison
out_eager_sum = out_eager.sum()
out_compiled_sum = out_compiled.sum()
diff = (out_eager_sum - out_compiled_sum).abs().item()
rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100
print(f'Relative diff (sum): {rel_diff:.6f}%')

if rel_diff > 5 and diff > 1:
    print(f' Forward output sums differ significantly (relative and absolute)!')
    print('out_eager_sum:', out_eager_sum.item())
    print('out_compiled_sum:', out_compiled_sum.item())
    print('Absolute diff:', diff)
    print('Relative diff (%):', rel_diff)
    sys.exit(1)