```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# No direct equivalent in TensorFlow, tf.function handles scalar outputs by default.

tf.random.set_seed(1352030645)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, sentinel):
    var_node_4 = arg_0 # size=(4, 8), dtype=bfloat16
    # Conversion: torch.full -> tf.fill with explicit dtype cast
    var_node_5 = tf.fill((8, 7), tf.cast(-0.80078125, tf.bfloat16)) # size=(8, 7), dtype=bfloat16
    var_node_3 = tf.matmul(var_node_4, var_node_5) # size=(4, 7), dtype=bfloat16
    var_node_7 = arg_1 # size=(7, 12), dtype=bfloat16
    var_node_8 = arg_2 # size=(12, 2), dtype=bfloat16
    var_node_6 = tf.matmul(var_node_7, var_node_8) # size=(7, 2), dtype=bfloat16
    var_node_2 = tf.matmul(var_node_3, var_node_6) # size=(4, 2), dtype=bfloat16
    var_node_11 = tf.fill((2, 3), tf.cast(1.515625, tf.bfloat16)) # size=(2, 3), dtype=bfloat16
    var_node_12 = tf.fill((3, 16), tf.cast(0.2353515625, tf.bfloat16)) # size=(3, 16), dtype=bfloat16
    var_node_10 = tf.matmul(var_node_11, var_node_12) # size=(2, 16), dtype=bfloat16
    var_node_14 = tf.fill((16, 4), tf.cast(2.21875, tf.bfloat16)) # size=(16, 4), dtype=bfloat16
    var_node_15 = tf.fill((4, 9), tf.cast(-1.7421875, tf.bfloat16)) # size=(4, 9), dtype=bfloat16
    var_node_13 = tf.matmul(var_node_14, var_node_15) # size=(16, 9), dtype=bfloat16
    var_node_9 = tf.matmul(var_node_10, var_node_13) # size=(2, 9), dtype=bfloat16
    var_node_1 = tf.matmul(var_node_2, var_node_9) # size=(4, 9), dtype=bfloat16
    var_node_19 = arg_3 # size=(14, 9), dtype=bfloat16
    var_node_20 = tf.fill((9, 2), tf.cast(0.8203125, tf.bfloat16)) # size=(9, 2), dtype=bfloat16
    var_node_18 = tf.matmul(var_node_19, var_node_20) # size=(14, 2), dtype=bfloat16
    var_node_22 = arg_4 # size=(2,), dtype=bfloat16
    var_node_23 = tf.fill((2,), tf.cast(-0.7421875, tf.bfloat16)) # size=(2,), dtype=bfloat16
    var_node_21 = tf.add(var_node_22, var_node_23) # size=(2,), dtype=bfloat16
    var_node_17 = tf.matmul(var_node_18, var_node_21) # size=(14,), dtype=bfloat16
    var_node_26 = arg_5 # size=(478, 13), dtype=bfloat16
    var_node_27 = tf.fill((13, 9), tf.cast(0.3359375, tf.bfloat16)) # size=(13, 9), dtype=bfloat16
    var_node_25 = tf.matmul(var_node_26, var_node_27) # size=(478, 9), dtype=bfloat16
    var_node_29 = arg_6 # size=(14,), dtype=int64
    var_node_30 = arg_7 # size=(14,), dtype=int64
    # Conversion: torch.div (int64) -> tf.math.floordiv
    var_node_28 = tf.math.floordiv(var_node_29, var_node_30) # size=(14,), dtype=int64
    # Conversion: torch.clamp -> tf.clip_by_value
    # Conversion: torch.nn.functional.embedding -> tf.gather
    var_node_28_clamped = tf.clip_by_value(var_node_28, 0, tf.shape(var_node_25)[0] - 1)
    var_node_24 = tf.gather(var_node_25, var_node_28_clamped) # size=(14, 9), dtype=bfloat16
    var_node_16 = tf.matmul(var_node_17, var_node_24) # size=(9,), dtype=bfloat16
    var_node_0 = tf.matmul(var_node_1, var_node_16) # size=(4,), dtype=bfloat16
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # Conversion: Python if -> tf.cond for graph compatibility
    result = tf.cond(tf.math.is_complex(result), lambda: tf.math.real(result), lambda: result)
    return result

# Sentinel tensor to ensure gradient computation
# Conversion: torch.tensor(..., requires_grad=True) -> tf.Variable(..., trainable=True)
sentinel = tf.Variable(1.0, dtype=tf.float32)

# Conversion: torch.as_strided + torch.randn -> tf.reshape + tf.random.normal
# The stride arguments in the source code correspond to standard contiguous memory layouts,
# so tf.reshape is the correct translation.
arg_0 = tf.reshape(tf.random.normal((32,), dtype=tf.bfloat16), (4, 8))
arg_1 = tf.reshape(tf.random.normal((84,), dtype=tf.bfloat16), (7, 12))
arg_2 = tf.reshape(tf.random.normal((24,), dtype=tf.bfloat16), (12, 2))
arg_3 = tf.reshape(tf.random.normal((126,), dtype=tf.bfloat16), (14, 9))
arg_4 = tf.reshape(tf.random.normal((2,), dtype=tf.bfloat16), (2,))
arg_5 = tf.reshape(tf.random.normal((6214,), dtype=tf.bfloat16), (478, 13))
# Conversion: torch.as_strided + torch.randint -> tf.reshape + tf.random.uniform
arg_6 = tf.reshape(tf.random.uniform((14,), minval=5, maxval=30, dtype=tf.int64), (14,))
arg_7 = tf.reshape(tf.random.uniform((14,), minval=5, maxval=30, dtype=tf.int64), (14,))

args = (arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')
# Conversion: torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```