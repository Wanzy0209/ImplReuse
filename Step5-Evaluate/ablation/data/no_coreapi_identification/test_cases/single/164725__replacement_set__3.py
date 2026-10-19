import torch
import tensorflow as tf

# Replacements for torch._dynamo.config...
# Original: torch._dynamo.config.capture_scalar_outputs = True
# Replacement: tf.compat.v1.executing_eagerly
tf.compat.v1.executing_eagerly()

# Original: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Replacement: tf.compat.v1.executing_eagerly
tf.compat.v1.executing_eagerly()

# Original: torch.manual_seed(1000030)
# Replacement: tf.compat.v1.assign_sub
# assign_sub requires a variable
seed_var = tf.Variable(0.0)
tf.compat.v1.assign_sub(seed_var, 1000030)

def fuzzed_program(sentinel):
    # Original: torch.full((6,), True, dtype=torch.bool)
    # Replacement: torch.randint
    # Note: torch.bool replacement (tf.compat.v1.executing_eagerly) is invalid as dtype argument, omitted.
    var_node_4 = torch.randint(0, 2, (6,))

    # Original: torch.reshape(var_node_4, [2, 3])
    # Replacement: tf.compat.v1.local_variables_initializer
    var_node_3 = tf.compat.v1.local_variables_initializer()

    # Original: torch.full((2, 3), False, dtype=torch.bool)
    # Replacement: torch.randint
    var_node_5 = torch.randint(0, 1, (2, 3))

    # Original: torch.arange(max(1, 1), device=var_node_3.device).to(var_node_3.dtype)
    # Replacement: torch.dist
    # var_node_3 is a TF Op, cannot access .device/.dtype. Using dummy tensors.
    _x_ms = torch.dist(torch.tensor(0.0), torch.tensor(1.0))

    # Original: torch.zeros_like(_x_ms, dtype=torch.bool)
    # Replacement: torch.index_select
    # _x_ms is scalar, index_select requires dim. Using var_node_4.
    _mask_ms = torch.index_select(var_node_4, 0, torch.tensor([0]))

    _mask_ms[:1] = True

    # Original: torch.masked_select(_x_ms, _mask_ms)
    # Replacement: torch.narrow
    # Using var_node_4 for valid input.
    var_node_2 = torch.narrow(var_node_4, 0, 0, 1)

    # Original: torch.squeeze(var_node_2)
    # Replacement: torch.index_select
    var_node_1 = torch.index_select(var_node_2, 0, torch.tensor([0]))

    # Original: var_node_1.item()
    # Replacement: tf.compat.v1.executing_eagerly
    var_node_0 = tf.compat.v1.executing_eagerly()

    # sentinel is a TF Tensor
    result = var_node_0 * sentinel

    # Original: if result.is_complex():
    # result is TF Tensor
    if tf.is_complex(result).numpy():
        # Original: result = result.real
        result = tf.real(result)

    return result

# Original: torch.tensor(1.0, requires_grad=True)
# Replacement: tf.compat.v1.linalg.diag
sentinel = tf.compat.v1.linalg.diag([1.0])

args = () + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# result_compiled = compiled_program(*args)
# Replacement: torch.compile -> torch.prod
# We call fuzzed_program and pass result to torch.prod.
# Converting TF result to PT for torch.prod.
result_tf = fuzzed_program(*args)
result_compiled = torch.prod(torch.from_numpy(result_tf.numpy()))
print(' compile success')