```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# Note: PyTorch Dynamo specific configuration, not applicable in TensorFlow

tf.random.set_seed(1061983224)

def fuzzed_program(arg_0, sentinel):
    var_node_4 = arg_0 # size=(4,), dtype=bool
    # Conversion: torch.chunk -> tf.split
    var_node_3 = tf.split(var_node_4, 4, axis=0)[0] # size=(1,), dtype=bool
    # Conversion: torch.squeeze -> tf.squeeze
    var_node_2 = tf.squeeze(var_node_3) # size=(), dtype=bool
    # Conversion: torch.stack -> tf.stack
    var_node_1 = tf.stack([var_node_2], axis=0) # size=(1,), dtype=bool
    # Conversion: torch.reshape -> tf.reshape
    var_node_0 = tf.reshape(var_node_1, [1]) # size=(1,), dtype=bool
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # Conversion: result.is_complex() -> tf.math.is_complex
    if tf.math.is_complex(result):
        # Conversion: result.real -> tf.math.real
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# Conversion: torch.tensor(..., requires_grad=True) -> tf.Variable
sentinel = tf.Variable(1.0)

# Conversion: torch.randint -> tf.random.uniform (for integer generation)
# Conversion: .bool() -> tf.cast(..., tf.bool)
# Conversion: torch.as_strided -> tf.reshape (for contiguous 1D stride)
arg_0 = tf.reshape(
    tf.cast(
        tf.random.uniform(shape=(4,), minval=0, maxval=2, dtype=tf.int8), 
        tf.bool
    ), 
    (4,)
)

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')

# Conversion: torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```