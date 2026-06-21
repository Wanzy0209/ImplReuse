import torch
import sys

# Fix: Check if torch._dynamo exists before accessing it to handle older PyTorch versions
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(238)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = torch.full((), 1, dtype=torch.int16)
    var_node_3 = arg_0
    var_node_1 = torch.add(var_node_2, var_node_3)
    var_node_5 = torch.full((1,), 3, dtype=torch.int16)
    var_node_4 = torch.squeeze(var_node_5)
    # Adapted call site: Replaced torch.div with torch.unique
    # torch.unique takes one input, so we use var_node_1
    var_node_0 = torch.unique(var_node_1)
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.as_strided(torch.randn(1).to(torch.int16), (), ())

args = (arg_0,) + (sentinel,)
out_eager = fuzzed_program(*args)
out_eager.sum().backward()
print('Eager Success! ')

# Fix: Check if torch.compile is available (requires PyTorch 2.0+)
if not hasattr(torch, 'compile'):
    print("torch.compile is not available in this PyTorch version. Skipping compiled test.")
    sys.exit(0)

compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
out_compiled = compiled_program(*args)
out_compiled.sum().backward()
print('Compile Success! ')
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