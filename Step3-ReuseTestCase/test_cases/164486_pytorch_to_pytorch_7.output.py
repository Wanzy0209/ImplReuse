import torch
torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(238)

def fuzzed_program(arg_0, sentinel):
    # Adapted dtype to float32 to support torch.log on CPU
    var_node_2 = torch.full((), 1, dtype=torch.float32)
    var_node_3 = arg_0
    var_node_1 = torch.add(var_node_2, var_node_3)
    
    # Replaced torch.div with torch.log
    # Note: torch.log is unary, so the second operand (var_node_4) is removed
    var_node_0 = torch.log(var_node_1)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Adapted dtype to float32
arg_0 = torch.as_strided(torch.randn(1).to(torch.float32), (), ())

args = (arg_0,) + (sentinel,)
out_eager = fuzzed_program(*args)
out_eager.sum().backward()
print('Eager Success! ')

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
    import sys; sys.exit(1)