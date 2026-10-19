import torch
import tensorflow as tf

# Replacements for torch._dynamo.config
# Original: torch._dynamo.config.capture_scalar_outputs = True
# Original: torch._dynamo.config.capture_dynamic_output_shape_ops = True
tf.compat.v1.experimental.output_all_intermediates = True
tf.compat.v1.experimental.output_all_intermediates = True

# Replacement for torch.manual_seed
# Original: torch.manual_seed(9)
# tf.compat.v1.assign requires a variable reference and a value.
# We create a dummy variable to ensure the call is runnable.
dummy_seed_var = tf.Variable(0)
tf.compat.v1.assign(dummy_seed_var, 9)

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0
    
    # Replacement for torch.full -> torch.fmin
    # Original: torch.full((1, 2), -66, dtype=torch.int32)
    # torch.fmin expects tensors, so we wrap arguments to ensure runnability.
    var_node_5 = torch.fmin(torch.tensor((1, 2)), torch.tensor(-66), dtype=tf.compat.v1.experimental.output_all_intermediates)
    
    # Original: torch.full((1, 2), 77, dtype=torch.int64)
    var_node_6 = torch.fmin(torch.tensor((1, 2)), torch.tensor(77), dtype=tf.compat.v1.experimental.output_all_intermediates)
    
    # Replacement for torch.ops.aten.add -> tf.compat.v1.experimental.output_all_intermediates
    var_node_4 = tf.compat.v1.experimental.output_all_intermediates(var_node_5, var_node_6)
    
    # Original: torch.full((1, 2), -64, dtype=torch.int32)
    var_node_7 = torch.fmin(torch.tensor((1, 2)), torch.tensor(-64), dtype=tf.compat.v1.experimental.output_all_intermediates)
    
    # Replacement for torch.ops.aten.mul -> tf.compat.v1.experimental.output_all_intermediates
    var_node_3 = tf.compat.v1.experimental.output_all_intermediates(var_node_4, var_node_7)
    
    # Original: torch.full((3, 4), False, dtype=torch.bool)
    var_node_9 = torch.fmin(torch.tensor((3, 4)), torch.tensor(False), dtype=tf.compat.v1.experimental.output_all_intermediates)
    
    # Replacement for torch.nonzero -> tf.compat.v1.experimental.output_all_intermediates
    var_node_8 = tf.compat.v1.experimental.output_all_intermediates(var_node_9)
    
    # Replacement for torch.ops.aten.add -> tf.compat.v1.experimental.output_all_intermediates
    var_node_2 = tf.compat.v1.experimental.output_all_intermediates(var_node_3, var_node_8)
    
    # Replacement for torch.ops.aten.div -> tf.compat.v1.experimental.output_all_intermediates
    var_node_0 = tf.compat.v1.experimental.output_all_intermediates(var_node_1, var_node_2)
    
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Replacement for torch.tensor -> torch.isinf
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# torch.isinf does not accept requires_grad, removed for runnability.
sentinel = torch.isinf(1.0)

# Replacement for torch.randint -> torch.amin
# Original: arg_0 = torch.randint(0, 3, (1, 2), dtype=torch.int64)
# torch.amin expects (input, dim). We map 0->input, 3->dim. 
# The size tuple (1, 2) is removed as it doesn't fit the signature.
arg_0 = torch.amin(torch.tensor(0), 3, dtype=tf.compat.v1.experimental.output_all_intermediates)

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

# Replacement for torch.compile -> torch.distributed.reduce
# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# torch.distributed.reduce expects (tensor, dst, ...). 
# We pass the function as the tensor argument to adhere to the replacement.
# We initialize a process group to allow the distributed call to proceed without crashing immediately.
try:
    torch.distributed.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:12345', world_size=1, rank=0)
except Exception as e:
    print(f"Init process group failed (expected if not in distributed setup): {e}")

# Note: fullgraph and dynamic are not valid args for reduce, removed for runnability.
compiled_program = torch.distributed.reduce(fuzzed_program, 0)

# The logic below attempts to call the result of reduce.
# Since reduce returns None/Future and not a compiled function, this is expected to fail,
# but we include it to preserve the original test case structure.
try:
    result_compiled = compiled_program(*args)
    print(' compile success with unbacked operations')
except Exception as e:
    print(f' compile failed (expected due to API replacement): {e}')

print(f'Eager result: {result_original}')