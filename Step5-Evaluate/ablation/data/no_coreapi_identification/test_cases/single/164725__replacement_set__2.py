import torch
import tensorflow as tf
import torch.distributed

# Replacements applied based on the provided mapping:
# - torch._dynamo.config.capture_scalar_outputs -> tf.compat.v1.global_variables_initializer
# - torch._dynamo.config.capture_dynamic_output_shape_ops -> tf.compat.v1.global_variables_initializer
# - torch.manual_seed -> tf.compat.v1.assign
# - torch.full -> torch.fmin
# - torch.bool -> tf.compat.v1.global_variables_initializer
# - torch.reshape -> tf.reverse
# - torch.arange -> torch.std
# - torch.zeros_like -> torch.nn.LazyLinear
# - torch.masked_select -> torch.isinf
# - torch.squeeze -> torch.nn.LazyLinear
# - item -> tf.compat.v1.global_variables_initializer
# - torch.tensor -> torch.isinf
# - torch.compile -> torch.distributed.reduce

# Setup for distributed (required for torch.distributed.reduce replacement)
try:
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', rank=0, world_size=1)
except Exception as e:
    print(f"Distributed init skipped or failed: {e}")

# Original: torch._dynamo.config.capture_scalar_outputs = True
# Replacement: tf.compat.v1.global_variables_initializer
tf.compat.v1.global_variables_initializer()

# Original: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Replacement: tf.compat.v1.global_variables_initializer
tf.compat.v1.global_variables_initializer()

# Original: torch.manual_seed(1000030)
# Replacement: tf.compat.v1.assign
# We create a dummy variable to assign to, as assign requires a ref.
dummy_var = tf.Variable(0)
tf.compat.v1.assign(dummy_var, 1000030)

def fuzzed_program(sentinel):
    # Original: var_node_4 = torch.full((6,), True, dtype=torch.bool)
    # Replacement: torch.full -> torch.fmin, torch.bool -> tf.compat.v1.global_variables_initializer
    # Note: torch.fmin does not strictly take a 'dtype' kwarg in this context, but we apply the replacement.
    var_node_4 = torch.fmin(torch.tensor(6), torch.tensor(True), dtype=tf.compat.v1.global_variables_initializer)

    # Original: var_node_3 = torch.reshape(var_node_4, [2, 3])
    # Replacement: tf.reverse
    var_node_3 = tf.reverse(var_node_4, [2, 3])

    # Original: var_node_5 = torch.full((2, 3), False, dtype=torch.bool)
    # Replacement: torch.fmin
    var_node_5 = torch.fmin(torch.tensor([2, 3]), torch.tensor(False), dtype=tf.compat.v1.global_variables_initializer)

    # Original: _x_ms = torch.arange(max(1, 1), device=var_node_3.device).to(var_node_3.dtype)
    # Replacement: torch.arange -> torch.std
    _x_ms = torch.std(torch.tensor(max(1, 1)), device=var_node_3.device).to(var_node_3.dtype)

    # Original: _mask_ms = torch.zeros_like(_x_ms, dtype=torch.bool)
    # Replacement: torch.zeros_like -> torch.nn.LazyLinear
    # LazyLinear is a module, so we instantiate and call it.
    layer_zeros = torch.nn.LazyLinear(1)
    _mask_ms = layer_zeros(_x_ms)

    # Original: _mask_ms[:1] = True
    # Keeping logic, though _mask_ms is now the output of a Linear layer.
    try:
        _mask_ms[:1] = True
    except Exception:
        pass # Ignore shape/type mismatches due to replacements

    # Original: var_node_2 = torch.masked_select(_x_ms, _mask_ms)
    # Replacement: torch.masked_select -> torch.isinf
    var_node_2 = torch.isinf(_x_ms)

    # Original: var_node_1 = torch.squeeze(var_node_2)
    # Replacement: torch.squeeze -> torch.nn.LazyLinear
    layer_squeeze = torch.nn.LazyLinear(1)
    var_node_1 = layer_squeeze(var_node_2)

    # Original: var_node_0 = var_node_1.item()
    # Replacement: item -> tf.compat.v1.global_variables_initializer
    var_node_0 = tf.compat.v1.global_variables_initializer()

    # Ensure gradient computation by multiplying with sentinel and taking real part
    # Note: var_node_0 is now a TF Op, not a scalar. This logic is preserved but will likely fail or behave differently.
    try:
        result = var_node_0 * sentinel
        if result.is_complex():
            result = result.real
    except Exception:
        result = sentinel # Fallback to ensure test continues
    return result

# Sentinel tensor to ensure gradient computation
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Replacement: torch.tensor -> torch.isinf
sentinel = torch.isinf(torch.tensor(1.0))

args = () + (sentinel,)

try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Replacement: torch.compile -> torch.distributed.reduce
# Note: reduce returns None/Tensor, not a callable function.
try:
    compiled_program = torch.distributed.reduce(sentinel, dst=0, op=torch.distributed.ReduceOp.SUM)
    
    # Original: result_compiled = compiled_program(*args)
    # This line will fail because compiled_program is not a function.
    result_compiled = compiled_program(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')