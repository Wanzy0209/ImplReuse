import torch
import sys

# Check for CUDA availability as the original bug report context implies CUDA usage
if not torch.cuda.is_available():
    print("CUDA is not available. Skipping test.")
    sys.exit(0)

torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(238)

def fuzzed_program(arg_0, sentinel):
    # Setup inputs similar to the original bug report
    var_node_2 = torch.full((), 1, dtype=torch.int16, device='cuda')
    var_node_3 = arg_0
    var_node_1 = torch.add(var_node_2, var_node_3)
    
    var_node_5 = torch.full((1,), 3, dtype=torch.int16, device='cuda')
    var_node_4 = torch.squeeze(var_node_5)
    
    # Adaptation: Replace torch.div with torch.add to test the similar API
    var_node_0 = torch.add(var_node_1, var_node_4)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Create input argument on CUDA
arg_0 = torch.as_strided(torch.randn(1, device='cuda').to(torch.int16), (), ())

args = (arg_0,) + (sentinel,)

# Run Eager Mode
out_eager = fuzzed_program(*args)
out_eager.sum().backward()
print('Eager Success! ')

# Run Compiled Mode
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
out_compiled = compiled_program(*args)
out_compiled.sum().backward()
print('Compile Success! ')

# Verification
out_eager_sum = out_eager.sum()
out_compiled_sum = out_compiled.sum()
diff = (out_eager_sum - out_compiled_sum).abs().item()
rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100

print(f'Relative diff (sum): {rel_diff:.6f}%')

# Assert that the outputs are consistent
if rel_diff > 5 and diff > 1:
    print(f' Forward output sums differ significantly (relative and absolute)!')
    print('out_eager_sum:', out_eager_sum.item())
    print('out_compiled_sum:', out_compiled_sum.item())
    print('Absolute diff:', diff)
    print('Relative diff (%):', rel_diff)
    sys.exit(1)
else:
    print('Test Passed! ')