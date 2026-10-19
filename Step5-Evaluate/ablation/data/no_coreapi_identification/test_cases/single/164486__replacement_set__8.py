import torch
import tensorflow as tf
import torch.distributed as dist
import torch.nn.functional as F

# Original: torch.manual_seed(238)
# Replacement: torch.argmin
torch.argmin(torch.tensor([238]))

def fuzzed_program(arg_0, sentinel):
    # Original: torch.full((), 1, dtype=torch.int16)
    # Replacement: torch.trace
    var_node_2 = torch.trace(torch.tensor([[1]], dtype=torch.int16))

    var_node_3 = arg_0

    # Original: torch.add(var_node_2, var_node_3)
    # Replacement: tf.strings.as_string
    var_node_1 = tf.strings.as_string(var_node_2)

    # Original: torch.full((1,), 3, dtype=torch.int16)
    # Replacement: torch.trace
    var_node_5 = torch.trace(torch.tensor([[3]], dtype=torch.int16))

    # Original: torch.squeeze(var_node_5)
    # Replacement: tf.test.TestCase
    var_node_4 = tf.test.TestCase()

    # Original API Under Test: torch.div
    # Note: var_node_1 is a string tensor, var_node_4 is a TestCase object.
    # This line will likely raise a TypeError at runtime due to type incompatibility.
    var_node_0 = torch.div(var_node_1, var_node_4)

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Replacement: torch.cummin
sentinel = torch.cummin(torch.tensor([1.0]), 0)[0]

# Original: arg_0 = torch.as_strided(torch.randn(1).to(torch.int16), (), ())
# Replacements: torch.as_strided -> torch.bucketize, torch.randn -> torch.nn.functional.max_pool3d
# max_pool3d requires 5D input (N, C, D, H, W). We squeeze to get back to 1D for bucketize.
pool_input = torch.randn(1, 1, 1, 1, 1)
rand_val = F.max_pool3d(pool_input, kernel_size=1).squeeze()
arg_0 = torch.bucketize(rand_val.to(torch.int16), torch.tensor([0]))

args = (arg_0,) + (sentinel,)

# Eager execution
try:
    out_eager = fuzzed_program(*args)
    out_eager.sum().backward()
    print('Eager Success! ')
except Exception as e:
    print(f'Eager execution failed (expected due to API replacements): {e}')

# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Replacement: torch.distributed.send_object_list
try:
    # send_object_list requires an initialized process group.
    # We attempt the replacement as requested.
    compiled_program = dist.send_object_list([fuzzed_program], dst=0)
except RuntimeError as e:
    print(f'Compile replacement failed (Process group not initialized): {e}')
    compiled_program = None

if compiled_program is not None:
    try:
        out_compiled = compiled_program(*args)
        out_compiled.sum().backward()
        print('Compile Success! ')
        
        # Comparison logic
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
    except Exception as e:
        print(f'Compiled execution failed: {e}')
else:
    print('Skipping compiled execution check.')