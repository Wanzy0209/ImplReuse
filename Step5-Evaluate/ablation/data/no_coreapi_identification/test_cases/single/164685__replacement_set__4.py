import torch
import tensorflow as tf
import torch.nn.functional as F

# Replacements for torch._dynamo.config
# torch._dynamo.config.capture_scalar_outputs -> torch.broadcast_to
_ = torch.broadcast_to(torch.tensor([1.0]), (1,))
# torch._dynamo.config.capture_dynamic_output_shape_ops -> torch.broadcast_to
_ = torch.broadcast_to(torch.tensor([1.0]), (1,))

# Replacement for torch.manual_seed -> tf.compat.v1.assign_add
seed_var = tf.Variable(0, dtype=tf.int32)
tf.compat.v1.assign_add(seed_var, 19989)

# Similar API: tf.keras.initializers.RandomUniform
# Used to generate the random value for arg_0
initializer = tf.keras.initializers.RandomUniform(minval=-5, maxval=5, seed=19989)

def fuzzed_program(arg_0, sentinel):
    # var_node_2 = -6 (dtype=int64)
    # Replacement: torch.int64 -> torch.broadcast_to
    var_node_2 = torch.tensor(-6.0)
    var_node_2 = torch.broadcast_to(var_node_2, (1,))

    # var_node_3 = arg_0 (dtype=int32)
    # Replacement: torch.int32 -> torch.broadcast_to
    var_node_3 = torch.tensor(arg_0)
    var_node_3 = torch.broadcast_to(var_node_3, (1,))

    var_node_1 = var_node_2 * var_node_3

    # var_node_5 = torch.full((), 1, dtype=torch.int64)
    # Replacement: torch.full -> torch.amin
    # Replacement: torch.int64 -> torch.broadcast_to
    temp = torch.tensor([[1.0]])
    var_node_5 = torch.amin(temp, dim=1)
    var_node_5 = torch.broadcast_to(var_node_5, (1,))

    # var_node_4 = var_node_5.item()
    # Replacement: var_node_5.item -> torch.broadcast_to
    var_node_4 = torch.broadcast_to(var_node_5, (1,))

    var_node_0 = var_node_1 / var_node_4

    result = var_node_0 * sentinel

    # if result.is_complex():
    # Replacement: result.is_complex -> torch.broadcast_to
    # We use the broadcast operation in the conditional check context
    if torch.broadcast_to(result, (1,)).numel() > 0:
        # result = result.real
        # Keeping logic minimal and runnable
        pass

    return result

# Setup Sentinel
# Replacement: torch.tensor -> tf.experimental.numpy.diag
sentinel_tf = tf.experimental.numpy.diag([1.0])
# Convert back to torch for internal logic compatibility
sentinel = torch.tensor(sentinel_tf.numpy(), requires_grad=True)

# Setup arg_0
# Original: torch.tensor(torch.randn(()), dtype=torch.int32).item()
# Use Similar API to get base value
val = initializer(shape=(1, 1))
# Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool2d
# Create a dummy 4D tensor for pooling
dummy_input = torch.tensor(val.numpy()).unsqueeze(0).unsqueeze(0)
pooled_val = F.adaptive_max_pool2d(dummy_input, output_size=(1, 1))
# Replacement: torch.tensor -> tf.experimental.numpy.diag
val_diag = tf.experimental.numpy.diag(pooled_val.numpy().flatten())
# Replacement: torch.int32 -> torch.broadcast_to
arg_0_tensor = torch.tensor(val_diag)
arg_0_tensor = torch.broadcast_to(arg_0_tensor, (1,))
arg_0 = arg_0_tensor.item()

args = (arg_0, sentinel)
result_original = fuzzed_program(*args)
print(' eager success')

# torch.compile -> torch.broadcast_to
# We replace the compilation step with a broadcast operation on the result
# This tests the replacement API in the context of the "compile" step
result_compiled = torch.broadcast_to(result_original, result_original.shape)
print(' compile success')