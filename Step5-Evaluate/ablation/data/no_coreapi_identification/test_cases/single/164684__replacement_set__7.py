import torch
import tensorflow as tf
from torch.distributions import constraints

# Replacements for torch._dynamo.config.capture_scalar_outputs
# and torch._dynamo.config.capture_dynamic_output_shape_ops
# Note: This assigns a boolean to the function reference, effectively overwriting it.
tf.keras.metrics.serialize = True
tf.keras.metrics.serialize = True

# Replacement for torch.manual_seed
# torch.argmax returns a tensor, we just call it to satisfy the replacement.
torch.argmax(torch.tensor(19990))

def fuzzed_program(arg_0, sentinel):
    var_node_2 = arg_0
    # Replacement for torch.squeeze
    # torch.distributions.constraints.cat expects a list of constraints.
    # We pass var_node_2 (a tensor) in a list to satisfy syntax.
    var_node_1 = torch.distributions.constraints.cat([var_node_2])

    var_node_0 = var_node_1.item()

    # Replacement for torch.compile,aten.mul.Tensor
    # tf.keras.metrics.serialize takes a metric instance.
    # We pass var_node_0.
    result = tf.keras.metrics.serialize(var_node_0)

    if result.is_complex():
        result = result.real
    return result

# Replacement for torch.tensor
# torch.cummax returns (values, indices). We unpack to keep sentinel as a tensor-like object.
sentinel, _ = torch.cummax(torch.tensor(1.0, requires_grad=True), dim=0)

# Replacement for torch.randint
# torch.argmin takes (input, dim). We create a dummy input.
# We omit dtype because argmin doesn't support it, to maintain runnability.
arg_0 = torch.argmin(torch.tensor([0, 1]), dim=0) > 0

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

# Replacement for torch.compile
# torch.lobpcg takes (A, ...). We pass fuzzed_program.
# We remove unsupported kwargs (fullgraph, dynamic) to attempt execution.
compiled_program = torch.lobpcg(fuzzed_program)
result_compiled = compiled_program(*args)
print(' compile success')