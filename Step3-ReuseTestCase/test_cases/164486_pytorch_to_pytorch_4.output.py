import torch

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(238)

def fuzzed_program(arg_0, sentinel):
    # Setup tensors similar to the original context
    var_node_2 = torch.full((), 1, dtype=torch.int16)
    var_node_3 = arg_0
    var_node_1 = torch.add(var_node_2, var_node_3)

    # Create a scalar tensor via squeeze to test edge cases
    var_node_5 = torch.full((1,), 3, dtype=torch.int16)
    var_node_4 = torch.squeeze(var_node_5)

    # Replace torch.div with torch.square
    # Original: var_node_0 = torch.div(var_node_1, var_node_4)
    var_node_0 = torch.square(var_node_1)

    # Ensure gradient computation
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Create the specific input tensor
arg_0 = torch.as_strided(torch.randn(1).to(torch.int16), (), ())

args = (arg_0,) + (sentinel,)

# Run Eager
out_eager = fuzzed_program(*args)
out_eager.sum().backward()
print('Eager Success! ')

# Run Compiled
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
out_compiled = compiled_program(*args)
out_compiled.sum().backward()
print('Compile Success! ')

# Verification
out_eager_sum = out_eager.sum()
out_compiled_sum = out_compiled.sum()
diff = (out_eager_sum - out_compiled_sum).abs().item()

# Handle potential division by zero for relative diff
denominator = out_eager_sum.abs().item() + 1e-12
rel_diff = diff / denominator * 100

print(f'Relative diff (sum): {rel_diff:.6f}%')

if rel_diff > 5 and diff > 1:
    print(f' Forward output sums differ significantly (relative and absolute)!')
    print('out_eager_sum:', out_eager_sum.item())
    print('out_compiled_sum:', out_compiled_sum.item())
    print('Absolute diff:', diff)
    print('Relative diff (%):', rel_diff)
    import sys
    sys.exit(1)