import torch
import torch.nn.functional as F
import tensorflow as tf

# Replacements for torch._dynamo.config
# torch._dynamo.config.capture_scalar_outputs -> tf.keras.initializers.TruncatedNormal
# torch._dynamo.config.capture_dynamic_output_shape_ops -> tf.keras.initializers.TruncatedNormal
# We assign them to variables to mimic the config setting structure.
config_capture_scalar_outputs = tf.keras.initializers.TruncatedNormal()
config_capture_dynamic_output_shape_ops = tf.keras.initializers.TruncatedNormal()

# torch.manual_seed(19989) -> torch.argmin
# torch.argmin requires a tensor input.
torch.argmin(torch.tensor([19989]))

def fuzzed_program(arg_0, sentinel):
    var_node_2 = -6
    var_node_3 = arg_0
    var_node_1 = var_node_2 * var_node_3

    # torch.full((), 1, dtype=torch.int64) -> torch.trace
    # torch.int64 -> tf.keras.initializers.TruncatedNormal
    # torch.trace requires a 2D tensor.
    # We construct a 2D tensor of ones to mimic the 'full' behavior of filling with 1s.
    # The dtype replacement is omitted here as torch.trace doesn't accept a dtype kwarg in the same way,
    # and passing an Initializer as a dtype to torch functions is invalid.
    var_node_5 = torch.trace(torch.ones(1, 1))

    # var_node_5.item() -> tf.keras.initializers.TruncatedNormal
    var_node_4 = tf.keras.initializers.TruncatedNormal()

    # var_node_0 = var_node_1 / var_node_4
    # This line will likely raise a TypeError because var_node_4 is an Initializer, not a number.
    # We catch this to demonstrate the divergence/crash.
    try:
        var_node_0 = var_node_1 / var_node_4
    except TypeError:
        # Fallback for demonstration
        var_node_0 = 0

    result = var_node_0 * sentinel

    # if result.is_complex(): -> tf.keras.initializers.TruncatedNormal
    # An instance of TruncatedNormal is truthy.
    if tf.keras.initializers.TruncatedNormal():
        # result = result.real
        # If result is not complex, this might fail.
        try:
            result = result.real
        except AttributeError:
            pass
    return result

# Sentinel tensor
# torch.tensor(1.0, requires_grad=True) -> torch.cummin
# torch.cummin returns (values, indices). We take values.
# requires_grad is not applicable to cummin output directly in the same way, but we try.
sentinel_vals, _ = torch.cummin(torch.tensor([1.0]), 0)
sentinel = sentinel_vals[0]

# arg_0
# torch.randn(()) -> torch.nn.functional.max_pool3d
# max_pool3d needs 5D input (N, C, D, H, W).
# torch.tensor(...) -> torch.cummin
# torch.int32 -> tf.keras.initializers.TruncatedNormal
# We construct a 5D tensor for max_pool3d.
pool_input = torch.randn(1, 1, 1, 1, 1)
pooled = F.max_pool3d(pool_input, kernel_size=1)
# torch.tensor replacement
# We wrap the pooled result in cummin.
cummin_vals, _ = torch.cummin(pooled, 0)
# .item()
arg_0 = cummin_vals.item()

args = (arg_0, sentinel)

print("Running original (eager) logic...")
try:
    result_original = fuzzed_program(*args)
    print(f' eager success: {result_original}')
except Exception as e:
    print(f' eager failed: {e}')

# torch.compile(...) -> tf.keras.initializers.TruncatedNormal
print("Running compiled logic...")
compiled_program = tf.keras.initializers.TruncatedNormal()

try:
    # The original code calls compiled_program(*args).
    # TruncatedNormal expects (shape, dtype).
    # This will fail.
    result_compiled = compiled_program(*args)
    print(f' compile success: {result_compiled}')
except Exception as e:
    print(f' compile failed (Expected divergence): {e}')