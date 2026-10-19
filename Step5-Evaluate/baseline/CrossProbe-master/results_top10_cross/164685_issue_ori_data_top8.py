```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: TensorFlow's tf.function handles graph compilation automatically. 
# Specific dynamo configs do not have direct 1:1 equivalents.

tf.random.set_seed(19989)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = -6 # dtype=int64
    var_node_3 = arg_0 # dtype=int32
    var_node_1 = var_node_2 * var_node_3 # dtype=int32
    # Conversion: torch.full((), 1, dtype=torch.int64) -> tf.constant
    var_node_5 = tf.constant(1, dtype=tf.int64) # size=(), dtype=int64
    # Conversion: .item() -> .numpy() to extract scalar value
    var_node_4 = var_node_5.numpy() # dtype=int64
    var_node_0 = var_node_1 / var_node_4 # dtype=float64 (TF promotes int division to float)
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # Conversion: result.is_complex() -> check dtype
    if result.dtype == tf.complex64 or result.dtype == tf.complex128:
        # Conversion: result.real -> tf.math.real
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# Conversion: torch.tensor(..., requires_grad=True) -> tf.Variable
sentinel = tf.Variable(1.0, dtype=tf.float32)

# Conversion: torch.randn(()) -> tf.random.normal([])
# Conversion: .item() -> .numpy() and cast to int
arg_0 = int(tf.cast(tf.random.normal([]), tf.int32).numpy())

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')

# Conversion: torch.compile -> tf.function
# fullgraph=True and dynamic=True are default behaviors or handled via retracing in tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```