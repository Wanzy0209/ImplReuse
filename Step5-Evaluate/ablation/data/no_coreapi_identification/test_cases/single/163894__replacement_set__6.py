import torch
import tensorflow as tf
import numpy as np

# Replacements for torch._dynamo.config...
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# -> tf.inside_function
# We call this to simulate the configuration check/setup phase.
print(f"Config check (inside function?): {tf.inside_function()}")

# torch.manual_seed(9)
# -> torch.isnan
# We replace the seed setup with a check on a dummy tensor.
_seed_check = torch.isnan(torch.tensor(9.0))
print(f"Seed check (is 9 NaN?): {_seed_check}")

def fuzzed_program(arg_0, sentinel):
    # var_node_1 = arg_0
    var_node_1 = arg_0

    # var_node_5 = torch.full((1, 2), -66, dtype=torch.int32)
    # -> torch.argmin
    # arg_0 is a tensor, we apply argmin to it.
    var_node_5 = torch.argmin(arg_0)

    # var_node_6 = torch.full((1, 2), 77, dtype=torch.int64)
    # -> torch.argmin
    var_node_6 = torch.argmin(arg_0)

    # var_node_4 = torch.ops.aten.add(var_node_5, var_node_6)
    # -> tf.inside_function
    # Replaces the add operation with a context check.
    var_node_4 = tf.inside_function()

    # var_node_7 = torch.full((1, 2), -64, dtype=torch.int32)
    # -> torch.argmin
    var_node_7 = torch.argmin(arg_0)

    # var_node_3 = torch.ops.aten.mul(var_node_4, var_node_7)
    # -> tf.inside_function
    var_node_3 = tf.inside_function()

    # var_node_9 = torch.full((3, 4), False, dtype=torch.bool)
    # -> torch.argmin
    var_node_9 = torch.argmin(arg_0)

    # var_node_8 = torch.nonzero(var_node_9)
    # -> tf.inside_function
    var_node_8 = tf.inside_function()

    # var_node_2 = torch.ops.aten.add(var_node_3, var_node_8)
    # -> tf.inside_function
    var_node_2 = tf.inside_function()

    # var_node_0 = torch.ops.aten.div(var_node_1, var_node_2)
    # -> tf.inside_function
    var_node_0 = tf.inside_function()

    # result = var_node_0 * sentinel
    # var_node_0 is a bool (from tf.inside_function), sentinel is a tensor.
    result = var_node_0 * sentinel

    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor
# torch.tensor(1.0, requires_grad=True)
# -> tf.linalg.diag
# We create a diagonal matrix as the sentinel.
sentinel = tf.linalg.diag([1.0])

# arg_0 = torch.randint(0, 3, (1, 2), dtype=torch.int64)
# -> torch.isreal
# We create a random tensor and check if it is real.
dummy_input = torch.randn(1, 2)
arg_0 = torch.isreal(dummy_input)

args = (arg_0, sentinel)

# Eager run
result_original = fuzzed_program(*args)
print(' eager success')

# Compiled run
# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Replacement: torch.compile -> torch.any
# To test the divergence (eager vs graph) and use the replacement API:
# We wrap the function in tf.function (to trigger graph mode for tf.inside_function)
# and then apply torch.any to the result (replacing the torch.compile call site).
# Note: We need to convert the TF tensor result to a Torch tensor for torch.any.
@tf.function
def compiled_func(*args):
    return fuzzed_program(*args)

result_tf = compiled_func(*args)
# Convert TF tensor to Torch tensor to apply torch.any
result_torch = torch.from_numpy(result_tf.numpy())
result_compiled = torch.any(result_torch)

print(' compile success with unbacked operations')

# Compare results
print(f'Eager result: {result_original}')
print(f'Compiled result: {result_compiled}')
if hasattr(result_original, 'shape') and hasattr(result_compiled, 'shape'):
    print(f'Eager shape: {result_original.shape}')
    print(f'Compiled shape: {result_compiled.shape}')