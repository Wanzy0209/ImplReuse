```python
import tensorflow as tf

# PyTorch specific configs, not applicable in TensorFlow
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(19990)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = arg_0 # size=(1,), dtype=bool
    # Conversion: torch.squeeze -> tf.squeeze
    var_node_1 = tf.squeeze(var_node_2) # size=(), dtype=bool
    # Conversion: .item() extracts a Python scalar. In TF graph mode, we keep it as a tensor.
    var_node_0 = var_node_1 
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # Conversion: is_complex() -> dtype.is_complex
    if result.dtype.is_complex:
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# Conversion: requires_grad=True is implicit in TF when using GradientTape, 
# here we use a constant/variable.
sentinel = tf.constant(1.0)

# Conversion: torch.randint -> tf.random.uniform with casting
arg_0 = tf.cast(tf.random.uniform((1,), 0, 2, dtype=tf.int32) > 0, tf.bool)

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
# Conversion: torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```