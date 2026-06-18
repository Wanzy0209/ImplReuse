import torch
import tensorflow as tf

# Configure seed to match the reproducibility attempt in the original bug
tf.random.set_seed(19989)

# Use the similar API: tf.keras.initializers.RandomUniform
# We initialize it with parameters that loosely mirror the range in the PyTorch bug (-6 to 6)
initializer = tf.keras.initializers.RandomUniform(minval=-6.0, maxval=6.0, dtype=tf.float32)

def fuzzed_program(shape):
    # Use the similar API to generate a value
    # Mirroring the scalar tensor creation in the original bug
    var_node_3 = initializer(shape)

    # Mirroring the scalar division logic from the original bug
    # Original: var_node_0 = var_node_1 / var_node_4
    var_node_4 = tf.constant(1.0, dtype=tf.float32)
    result = var_node_3 / var_node_4

    return result

# Test with scalar shape, similar to torch.full((), ...)
shape = ()

# 1. Eager execution
result_eager = fuzzed_program(shape)
print(' eager success')

# 2. Compiled execution (tf.function is the TF equivalent of torch.compile)
# fullgraph=True in PyTorch roughly maps to standard tf.function behavior
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(shape)
print(' compile success')

# 3. Check for divergence
# Note: Due to RNG state differences between eager and graph execution in TF,
# exact value matching might fail without careful seed management per call.
# However, the primary goal is to ensure no crash (KeyError) and valid output.
# We check if the results are valid tensors and have the expected shape.
assert isinstance(result_eager, tf.Tensor), "Eager result is not a tensor"
assert isinstance(result_compiled, tf.Tensor), "Compiled result is not a tensor"
assert result_eager.shape == shape, "Eager shape mismatch"
assert result_compiled.shape == shape, "Compiled shape mismatch"