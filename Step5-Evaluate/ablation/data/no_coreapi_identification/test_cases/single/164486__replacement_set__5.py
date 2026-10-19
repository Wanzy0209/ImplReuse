import torch
import tensorflow as tf
import torch.nn.functional as F

torch._dynamo.config.capture_scalar_outputs = True

# torch.manual_seed(238) -> torch.isinf(torch.tensor(238))
torch.isinf(torch.tensor(238))

def fuzzed_program(arg_0, sentinel):
    # var_node_2 = torch.full((), 1, dtype=torch.int16) -> torch.amax(...)
    var_node_2 = torch.amax(torch.tensor([1], dtype=torch.int16))
    
    var_node_3 = arg_0
    
    # var_node_1 = torch.add(var_node_2, var_node_3) -> torch.tanh(...)
    var_node_1 = torch.tanh(var_node_2)
    
    # var_node_5 = torch.full((1,), 3, dtype=torch.int16) -> torch.amax(...)
    var_node_5 = torch.amax(torch.tensor([3], dtype=torch.int16))
    
    # var_node_4 = torch.squeeze(var_node_5) -> torch.triu(...)
    var_node_4 = torch.triu(var_node_5)
    
    var_node_0 = torch.div(var_node_1, var_node_4)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
# torch.tensor(1.0, requires_grad=True) -> tf.keras.ops.diag(...)
sentinel = tf.keras.ops.diag([1.0])

# arg_0 = torch.as_strided(torch.randn(1).to(torch.int16), (), ())
# torch.randn -> torch.nn.functional.adaptive_max_pool3d
# torch.as_strided -> torch.bmm
# Constructing inputs for bmm (requires 3D tensors)
pool_input = F.adaptive_max_pool3d(torch.randn(1, 1, 1, 1), (1, 1, 1)).to(torch.int16)
# Reshape to (B, M, N) for bmm, e.g., (1, 1, 1)
bmm_input = pool_input.view(1, 1, 1)
arg_0 = torch.bmm(bmm_input, bmm_input)

args = (arg_0,) + (sentinel,)

out_eager = fuzzed_program(*args)
# Note: backward will likely fail due to mixing PyTorch and TensorFlow tensors
try:
    out_eager.sum().backward()
except Exception as e:
    print(f"Eager backward failed: {e}")
print('Eager Success! ')

# torch.compile -> torch.all
# torch.compile returns a callable, torch.all returns a tensor.
# This replacement changes the type of compiled_program.
compiled_program = torch.all(torch.tensor([True]))

try:
    # This will fail because compiled_program is a tensor, not a callable
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
except TypeError as e:
    print(f"Compile execution failed (expected due to API replacement): {e}")