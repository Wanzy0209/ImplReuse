```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True # PyTorch specific config, omitted in TF
# torch._dynamo.config.capture_dynamic_output_shape_ops = True # PyTorch specific config, omitted in TF

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(19990)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = arg_0 # size=(1,), dtype=bool
    # Conversion: torch.squeeze -> tf.squeeze
    var_node_1 = tf.squeeze(var_node_2) # size=(), dtype=bool
    # Conversion: .item() extracts Python scalar, but in TF graph we keep as tensor
    var_node_0 = var_node_1
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # Conversion: Python if -> tf.cond for graph compatibility
    # Conversion: result.is_complex() -> tf.math.is_complex(result)
    # Conversion: result.real -> tf.math.real(result)
    result = tf.cond(tf.math.is_complex(result),
                     lambda: tf.math.real(result),
                     lambda: result)
    return result

# Sentinel tensor to ensure gradient computation
# Conversion: torch.tensor(..., requires_grad=True) -> tf.Variable
sentinel = tf.Variable(1.0, dtype=tf.float32)

# Conversion: torch.randint(0, 2, (1,), dtype=torch.bool) > 0
# Note: tf.random.uniform generates float by default, need int32 for the logic
arg_0 = tf.random.uniform(shape=(1,), minval=0, maxval=2, dtype=tf.int32) > 0

args = (arg_0, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')
# Conversion: torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```