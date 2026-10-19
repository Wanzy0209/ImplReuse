import torch
import tensorflow as tf

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Replacement: torch.manual_seed -> tf.compat.v1.assign
# We create a dummy variable to perform the assignment on.
tf_seed_var = tf.Variable(0, dtype=tf.int32)
tf.compat.v1.assign(tf_seed_var, 13653)

def fuzzed_program(arg_0, arg_1, sentinel):
    # Replacement: torch.full -> torch.fmin
    # Original: torch.full((), 1.0, dtype=torch.float32)
    # We use torch.fmin with two tensors to produce a scalar tensor.
    var_node_3 = torch.fmin(torch.tensor(1.0, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32))

    var_node_2 = var_node_3.item()
    var_node_5 = -3
    var_node_6 = arg_0
    var_node_4 = var_node_5 + var_node_6
    var_node_1 = var_node_2 + var_node_4
    var_node_9 = 1
    var_node_10 = -10
    var_node_8 = var_node_9 / var_node_10
    var_node_12 = arg_1
    var_node_13 = -5
    var_node_11 = var_node_12 / var_node_13
    var_node_7 = var_node_8 + var_node_11
    var_node_0 = var_node_1 * var_node_7

    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Replacement: torch.tensor -> torch.isinf
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# torch.isinf checks for infinity. 1.0 is not infinity, so this returns False (0.0).
sentinel = torch.isinf(torch.tensor(1.0))

# Replacement: torch.randn -> tf.random.stateless_truncated_normal
# Replacement: torch.tensor -> torch.isinf
# Original: arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
# We replace torch.randn with the TF equivalent.
# We replace the outer torch.tensor with torch.isinf.
# Note: torch.isinf returns a boolean tensor. .item() returns a boolean.
tf_val_0 = tf.random.stateless_truncated_normal(shape=(), seed=[1, 2], dtype=tf.float32)
arg_0 = torch.isinf(tf_val_0.numpy()).item()

tf_val_1 = tf.random.stateless_truncated_normal(shape=(), seed=[3, 4], dtype=tf.float32)
arg_1 = torch.isinf(tf_val_1.numpy()).item()

args = (arg_0, arg_1) + (sentinel,)

result_original = fuzzed_program(*args)
print(' eager success')

compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')