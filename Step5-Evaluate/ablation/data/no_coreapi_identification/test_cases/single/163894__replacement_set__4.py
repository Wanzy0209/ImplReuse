import torch
import tensorflow as tf

# Replacements for torch._dynamo.config
# torch._dynamo.config.capture_scalar_outputs -> torch.backends.cudnn.version
# torch._dynamo.config.capture_dynamic_output_shape_ops -> torch.backends.cudnn.version
torch.backends.cudnn.version = True
torch.backends.cudnn.version = True

# torch.manual_seed -> tf.compat.v1.assign_add
# Note: assign_add requires a variable ref. We pass the seed value blindly as per replacement.
tf.compat.v1.assign_add(9)

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0

    # torch.full -> torch.amin
    # torch.int32 -> torch.backends.cudnn.version
    var_node_5 = torch.amin((1, 2), -66, dtype=torch.backends.cudnn.version)

    # torch.full -> torch.amin
    # torch.int64 -> torch.backends.cudnn.version
    var_node_6 = torch.amin((1, 2), 77, dtype=torch.backends.cudnn.version)

    # torch.ops.aten.add -> torch.backends.cudnn.version
    var_node_4 = torch.backends.cudnn.version(var_node_5, var_node_6)

    # torch.full -> torch.amin
    var_node_7 = torch.amin((1, 2), -64, dtype=torch.backends.cudnn.version)

    # torch.ops.aten.mul -> torch.backends.cudnn.version
    var_node_3 = torch.backends.cudnn.version(var_node_4, var_node_7)

    # torch.full -> torch.amin
    # torch.bool -> torch.backends.cudnn.version
    var_node_9 = torch.amin((3, 4), False, dtype=torch.backends.cudnn.version)

    # torch.nonzero -> torch.backends.cudnn.version
    var_node_8 = torch.backends.cudnn.version(var_node_9)

    # torch.ops.aten.add -> torch.backends.cudnn.version
    var_node_2 = torch.backends.cudnn.version(var_node_3, var_node_8)

    # torch.ops.aten.div -> torch.backends.cudnn.version
    var_node_0 = torch.backends.cudnn.version(var_node_1, var_node_2)

    # result = var_node_0 * sentinel
    result = var_node_0 * sentinel

    # Leverage similar API: tf.compat.v1.control_flow_v2_enabled
    if tf.compat.v1.control_flow_v2_enabled():
        if result.is_complex():
            result = result.real

    return result

# torch.tensor -> tf.experimental.numpy.diag
sentinel = tf.experimental.numpy.diag(1.0, requires_grad=True)

# torch.randint -> torch.fmax
# torch.int64 -> torch.backends.cudnn.version
arg_0 = torch.fmax(0, 3, (1, 2), dtype=torch.backends.cudnn.version)

args = (arg_0,) + (sentinel,)

print("Testing eager execution...")
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
    print(f'Eager result: {result_original}')
except Exception as e:
    print(f' eager failed: {e}')

print("\nTesting compilation...")
try:
    # torch.compile -> torch.library.impl_abstract
    # Note: impl_abstract expects (func, name) or (name, func). The original compile call had (func, options).
    # We pass the arguments as they were.
    compiled_program = torch.library.impl_abstract(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success with unbacked operations')
    print(f'Compiled result: {result_compiled}')
except Exception as e:
    print(f' compile failed: {e}')