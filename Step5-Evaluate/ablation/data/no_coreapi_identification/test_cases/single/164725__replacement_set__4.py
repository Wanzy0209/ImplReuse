import torch
import tensorflow as tf
import torch.fx

# Replacements for torch._dynamo.config...
# Original: torch._dynamo.config.capture_scalar_outputs = True
# Replacement: tf.executing_eagerly()
print("Config check 1:", tf.executing_eagerly())

# Original: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Replacement: tf.executing_eagerly()
print("Config check 2:", tf.executing_eagerly())

# Original: torch.manual_seed(1000030)
# Replacement: tf.compat.v1.assign_add
# Create a variable to act as the seed storage
seed_var = tf.Variable(0.0)
tf.compat.v1.assign_add(seed_var, 1000030.0)

def fuzzed_program(sentinel):
    # Original: var_node_4 = torch.full((6,), True, dtype=torch.bool)
    # Replacement: torch.amin
    # Create a dummy tensor to apply amin
    dummy_1 = torch.randn(6, 1)
    var_node_4 = torch.amin(dummy_1, dim=1)

    # Original: var_node_3 = torch.reshape(var_node_4, [2, 3])
    # Replacement: tf.compat.v1.global_variables_initializer
    var_node_3 = tf.compat.v1.global_variables_initializer()

    # Original: var_node_5 = torch.full((2, 3), False, dtype=torch.bool)
    # Replacement: torch.amin
    dummy_2 = torch.randn(2, 3, 1)
    var_node_5 = torch.amin(dummy_2, dim=2)

    # Original: _x_ms = torch.arange(max(1, 1), device=var_node_3.device).to(var_node_3.dtype)
    # Replacement: torch.quantile
    # var_node_3 is a TF Op, so we ignore device/dtype logic
    _x_ms = torch.quantile(torch.randn(10), 0.5)

    # Original: _mask_ms = torch.zeros_like(_x_ms, dtype=torch.bool)
    # Replacement: torch.stack
    _mask_ms = torch.stack([_x_ms, _x_ms])

    # Original: _mask_ms[:1] = True
    _mask_ms[:1] = True

    # Original: var_node_2 = torch.masked_select(_x_ms, _mask_ms)
    # Replacement: tf.keras.backend.batch_set_value
    var_node_2 = tf.keras.backend.batch_set_value([])

    # Original: var_node_1 = torch.squeeze(var_node_2)
    # Replacement: torch.fx.wrap
    # Wrap a dummy function
    @torch.fx.wrap
    def wrap_fn(x):
        return x
    var_node_1 = wrap_fn(var_node_2)

    # Original: var_node_0 = var_node_1.item()
    # Replacement: tf.executing_eagerly()
    var_node_0 = tf.executing_eagerly()

    # Original: result = var_node_0 * sentinel
    result = var_node_0 * sentinel

    # Original: if result.is_complex():
    if result.is_complex():
        result = result.real
    return result

# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Replacement: tf.experimental.numpy.diag
sentinel = tf.experimental.numpy.diag([1.0])

args = () + (sentinel,)

# Original: result_original = fuzzed_program(*args)
result_original = fuzzed_program(*args)
print(' eager success')

# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Replacement: torch.library.impl_abstract
# We use a dummy schema name
compiled_program = torch.library.impl_abstract("test::fuzzed_op")(fuzzed_program)

# Original: result_compiled = compiled_program(*args)
# Note: impl_abstract registers the function, it doesn't return a callable wrapper in the same way compile does.
# However, we attempt to call it to satisfy the test structure.
# If it returns None, this will raise TypeError.
# To make it runnable/minimal as requested, we check if callable.
if callable(compiled_program):
    result_compiled = compiled_program(*args)
    print(' compile success')
else:
    print(' compiled_program is not callable (returned None from impl_abstract)')