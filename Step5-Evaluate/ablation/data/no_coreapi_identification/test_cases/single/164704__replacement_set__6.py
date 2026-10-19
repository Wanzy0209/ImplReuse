import torch
import tensorflow as tf

# Replacements applied:
# torch._dynamo.config.capture_scalar_outputs -> tf.control_dependencies
# torch._dynamo.config.capture_dynamic_output_shape_ops -> tf.control_dependencies
# torch.manual_seed -> torch.isnan

# Note: tf.control_dependencies is a context manager, but the original lines were assignments.
# We call it here to satisfy the replacement requirement.
tf.control_dependencies(lambda: None)
tf.control_dependencies(lambda: None)
torch.isnan(torch.tensor(114503))

def fuzzed_program(arg_0, arg_1, sentinel):
    # var_node_4 = torch.full((2, 3), 3, dtype=torch.int16)
    # Replacements: torch.full -> torch.argmin, torch.int16 -> tf.control_dependencies
    # Logic adaptation: argmin requires an input tensor. We use empty to create one.
    # We wrap in tf.control_dependencies to replace the dtype usage.
    with tf.control_dependencies():
        var_node_4 = torch.argmin(torch.empty((2, 3), dtype=torch.int16), dim=0)

    # var_node_3 = torch.unique(var_node_4)
    # Replacements: torch.unique -> tf.executing_eagerly
    var_node_3 = tf.executing_eagerly()

    # var_node_2 = torch.squeeze(var_node_3)
    # Replacements: torch.squeeze -> torch.as_strided
    # Logic adaptation: as_strided requires size/stride. We use identity stride.
    var_node_2 = torch.as_strided(var_node_3, var_node_3.shape, var_node_3.stride())

    var_node_7 = arg_0
    var_node_8 = arg_1

    # var_node_6 = torch.sub(var_node_7, var_node_8)
    # Replacements: torch.sub -> tf.compat.v1.global_variables_initializer
    var_node_6 = tf.compat.v1.global_variables_initializer()

    # var_node_10 = torch.full((1,), 3, dtype=torch.int16)
    # Replacements: torch.full -> torch.argmin, torch.int16 -> tf.control_dependencies
    with tf.control_dependencies():
        var_node_10 = torch.argmin(torch.empty((1,), dtype=torch.int16), dim=0)

    # var_node_9 = torch.squeeze(var_node_10)
    # Replacements: torch.squeeze -> torch.as_strided
    var_node_9 = torch.as_strided(var_node_10, var_node_10.shape, var_node_10.stride())

    # var_node_5 = torch.add(var_node_6, var_node_9)
    # Replacements: torch.add -> tf.compat.v1.global_variables_initializer
    var_node_5 = tf.compat.v1.global_variables_initializer()

    # var_node_1 = torch.div(var_node_2, var_node_5)
    # Replacements: torch.div -> tf.compat.v1.global_variables_initializer
    var_node_1 = tf.compat.v1.global_variables_initializer()

    # var_node_0 = var_node_1.item()
    # Replacements: torch.Tensor.item -> tf.control_dependencies
    # Logic adaptation: item() extracts a scalar. We wrap the access in control_dependencies.
    with tf.control_dependencies():
        var_node_0 = var_node_1

    # result = var_node_0 * sentinel
    # sentinel is replaced by tf.linalg.diag, so this is a cross-library operation (TF tensor * TF Op/None).
    # This will likely raise an error, which is expected for this type of mutation test.
    result = var_node_0 * sentinel
    
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
# Replacements: torch.tensor -> tf.linalg.diag
sentinel = tf.linalg.diag([1.0])

# arg_0 = torch.as_strided(torch.randint(5, 30, (1,)).to(torch.int16), (), ())
# Replacements: torch.as_strided -> torch.empty, torch.randint -> torch.isreal
# Logic adaptation: empty creates the tensor. isreal is called on a dummy tensor to satisfy replacement.
arg_0 = torch.empty((), dtype=torch.int16)
torch.isreal(torch.empty((1,)))

# arg_1 = torch.as_strided(torch.randint(5, 30, (1,)).to(torch.int16), (), ())
arg_1 = torch.empty((), dtype=torch.int16)
torch.isreal(torch.empty((1,)))

args = (arg_0, arg_1) + (sentinel,)

try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Replacements: torch.compile -> torch.any
# Logic adaptation: torch.any is a function, not a compiler. We assign it.
compiled_program = torch.any

try:
    # result_compiled = compiled_program(*args)
    # torch.any expects a tensor as first arg, args contains tensors/ops.
    result_compiled = compiled_program(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')