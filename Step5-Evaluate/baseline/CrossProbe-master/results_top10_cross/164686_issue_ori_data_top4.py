```python
import tensorflow as tf

# PyTorch: torch._dynamo.config.capture_scalar_outputs = True
# PyTorch: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# TensorFlow: tf.function handles scalars and dynamic shapes by default, no explicit config needed.

# PyTorch: torch.manual_seed(13653)
tf.random.set_seed(13653)

def fuzzed_program(arg_0, arg_1, sentinel):
    # PyTorch: var_node_3 = torch.full((), 1.0, dtype=torch.float32)
    var_node_3 = tf.fill((), 1.0) # dtype=float32

    # PyTorch: var_node_2 = var_node_3.item()
    # In TensorFlow, we keep it as a tensor to maintain graph compatibility.
    var_node_2 = var_node_3

    # PyTorch: var_node_5 = -3 # dtype=int32
    var_node_5 = tf.constant(-3, dtype=tf.int32)

    # PyTorch: var_node_6 = arg_0 # dtype=int64
    # arg_0 is passed as a Python int (from .item()), cast to int64 tensor to match type
    var_node_6 = tf.cast(arg_0, tf.int64)

    # PyTorch: var_node_4 = var_node_5 + var_node_6 # dtype=int32
    # Note: PyTorch promotes int32 + int64 to int64. 
    # We follow the operation logic.
    var_node_4 = var_node_5 + var_node_6

    # PyTorch: var_node_1 = var_node_2 + var_node_4 # dtype=float32
    var_node_1 = var_node_2 + tf.cast(var_node_4, tf.float32)

    # PyTorch: var_node_9 = 1 # dtype=int64
    var_node_9 = tf.constant(1, dtype=tf.int64)

    # PyTorch: var_node_10 = -10 # dtype=int32
    var_node_10 = tf.constant(-10, dtype=tf.int32)

    # PyTorch: var_node_8 = var_node_9 / var_node_10 # dtype=int64
    # Note: Division in PyTorch returns float. We follow the operation.
    var_node_8 = var_node_9 / var_node_10

    # PyTorch: var_node_12 = arg_1 # dtype=int64
    var_node_12 = tf.cast(arg_1, tf.int64)

    # PyTorch: var_node_13 = -5 # dtype=int32
    var_node_13 = tf.constant(-5, dtype=tf.int32)

    # PyTorch: var_node_11 = var_node_12 / var_node_13 # dtype=int32
    # Note: Division returns float.
    var_node_11 = var_node_12 / var_node_13

    # PyTorch: var_node_7 = var_node_8 + var_node_11 # dtype=int32
    # Note: Float + Float = Float.
    var_node_7 = var_node_8 + var_node_11

    # PyTorch: var_node_0 = var_node_1 * var_node_7 # dtype=float32
    var_node_0 = var_node_1 * var_node_7

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel

    # PyTorch: if result.is_complex():
    if result.dtype.is_complex:
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# PyTorch: sentinel = torch.tensor(1.0, requires_grad=True)
sentinel = tf.Variable(1.0, dtype=tf.float32)

# PyTorch: arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
# Generate random normal, cast to int64, extract scalar value
arg_0 = tf.cast(tf.random.normal(shape=()), tf.int64).numpy()

# PyTorch: arg_1 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
arg_1 = tf.cast(tf.random.normal(shape=()), tf.int64).numpy()

args = (arg_0, arg_1, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# PyTorch: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# TensorFlow: tf.function compiles the graph. dynamic=True is implicit.
compiled_program = tf.function(fuzzed_program)

result_compiled = compiled_program(*args)
print('✅ compile success')
```