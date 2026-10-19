```python
import tensorflow as tf
import numpy as np

# torch._dynamo.config.capture_scalar_outputs = True is specific to PyTorch Dynamo, not needed in TF

# Set random seed
tf.random.set_seed(1352030645)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, sentinel):
    var_node_4 = arg_0 # size=(4, 8), stride=(8, 1), dtype=bfloat16, device=cuda
    # Conversion: torch.full -> tf.constant
    var_node_5 = tf.constant(-0.80078125, shape=[8, 7], dtype=tf.bfloat16) # size=(8, 7), stride=(7, 1), dtype=bfloat16, device=cuda
    # Conversion: torch.matmul -> tf.matmul, .to() -> tf.cast
    var_node_3 = tf.matmul(tf.cast(var_node_4, tf.bfloat16), tf.cast(var_node_5, tf.bfloat16)) # size=(4, 7), stride=(7, 1), dtype=bfloat16, device=cuda
    var_node_7 = arg_1 # size=(7, 12), stride=(12, 1), dtype=bfloat16, device=cuda
    var_node_8 = arg_2 # size=(12, 2), stride=(2, 1), dtype=bfloat16, device=cuda
    var_node_6 = tf.matmul(tf.cast(var_node_7, tf.bfloat16), tf.cast(var_node_8, tf.bfloat16)) # size=(7, 2), stride=(2, 1), dtype=bfloat16, device=cuda
    var_node_2 = tf.matmul(tf.cast(var_node_3, tf.bfloat16), tf.cast(var_node_6, tf.bfloat16)) # size=(4, 2), stride=(2, 1), dtype=bfloat16, device=cuda
    var_node_11 = tf.constant(1.515625, shape=[2, 3], dtype=tf.bfloat16) # size=(2, 3), stride=(3, 1), dtype=bfloat16, device=cuda
    var_node_12 = tf.constant(0.2353515625, shape=[3, 16], dtype=tf.bfloat16) # size=(3, 16), stride=(16, 1), dtype=bfloat16, device=cuda
    var_node_10 = tf.matmul(tf.cast(var_node_11, tf.bfloat16), tf.cast(var_node_12, tf.bfloat16)) # size=(2, 16), stride=(16, 1), dtype=bfloat16, device=cuda
    var_node_14 = tf.constant(2.21875, shape=[16, 4], dtype=tf.bfloat16) # size=(16, 4), stride=(4, 1), dtype=bfloat16, device=cuda
    var_node_15 = tf.constant(-1.7421875, shape=[4, 9], dtype=tf.bfloat16) # size=(4, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_13 = tf.matmul(tf.cast(var_node_14, tf.bfloat16), tf.cast(var_node_15, tf.bfloat16)) # size=(16, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_9 = tf.matmul(tf.cast(var_node_10, tf.bfloat16), tf.cast(var_node_13, tf.bfloat16)) # size=(2, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_1 = tf.matmul(tf.cast(var_node_2, tf.bfloat16), tf.cast(var_node_9, tf.bfloat16)) # size=(4, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_19 = arg_3 # size=(14, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_20 = tf.constant(0.8203125, shape=[9, 2], dtype=tf.bfloat16) # size=(9, 2), stride=(2, 1), dtype=bfloat16, device=cuda
    var_node_18 = tf.matmul(tf.cast(var_node_19, tf.bfloat16), tf.cast(var_node_20, tf.bfloat16)) # size=(14, 2), stride=(2, 1), dtype=bfloat16, device=cuda
    var_node_22 = arg_4 # size=(2,), stride=(1,), dtype=bfloat16, device=cuda
    var_node_23 = tf.constant(-0.7421875, shape=[2], dtype=tf.bfloat16) # size=(2,), stride=(1,), dtype=bfloat16, device=cuda
    # Conversion: torch.add -> tf.add
    var_node_21 = tf.add(var_node_22, var_node_23) # size=(2,), stride=(1,), dtype=bfloat16, device=cuda
    var_node_17 = tf.matmul(tf.cast(var_node_18, tf.bfloat16), tf.cast(var_node_21, tf.bfloat16)) # size=(14,), stride=(1,), dtype=bfloat16, device=cuda
    var_node_26 = arg_5 # size=(478, 13), stride=(13, 1), dtype=bfloat16, device=cuda
    var_node_27 = tf.constant(0.3359375, shape=[13, 9], dtype=tf.bfloat16) # size=(13, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_25 = tf.matmul(tf.cast(var_node_26, tf.bfloat16), tf.cast(var_node_27, tf.bfloat16)) # size=(478, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_29 = arg_6 # size=(14,), stride=(1,), dtype=int64, device=cuda
    var_node_30 = arg_7 # size=(14,), stride=(1,), dtype=int64, device=cuda
    # Conversion: torch.div -> tf.math.divide
    var_node_28 = tf.math.divide(var_node_29, var_node_30) # size=(14,), stride=(1,), dtype=int64, device=cuda
    # Conversion: torch.nn.functional.embedding -> tf.nn.embedding_lookup
    # Note: embedding_lookup(params, ids) vs embedding(input, weight)
    # Also clamp -> clip_by_value, .size(0) -> tf.shape(...)[0]
    indices = tf.cast(tf.clip_by_value(var_node_28, 0, tf.shape(var_node_25)[0] - 1), tf.int64)
    var_node_24 = tf.nn.embedding_lookup(var_node_25, indices) # size=(14, 9), stride=(9, 1), dtype=bfloat16, device=cuda
    var_node_16 = tf.matmul(tf.cast(var_node_17, tf.bfloat16), tf.cast(var_node_24, tf.bfloat16)) # size=(9,), stride=(1,), dtype=bfloat16, device=cuda
    var_node_0 = tf.matmul(tf.cast(var_node_1, tf.bfloat16), tf.cast(var_node_16, tf.bfloat16)) # size=(4,), stride=(1,), dtype=bfloat16, device=cuda
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # Conversion: result.is_complex() -> result.dtype.is_complex
    if result.dtype.is_complex:
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# In TF, gradients are tracked via GradientTape, but for forward pass compatibility:
sentinel = tf.constant(1.0)

# Conversion: torch.as_strided -> tf.reshape (assuming contiguous data)
# Conversion: torch.randn -> tf.random.normal
# Conversion: torch.randint -> tf.random.uniform
arg_0 = tf.reshape(tf.random.normal([32], dtype=tf.bfloat16), [4, 8])
arg_1 = tf.reshape(tf.random.normal([84], dtype=tf.bfloat16), [7, 12])
arg_2 = tf.reshape(tf.random.normal([24], dtype=tf.bfloat16), [12, 2])
arg_3 = tf.reshape(tf.random.normal([126], dtype=tf.bfloat16), [14, 9])
arg_4 = tf.reshape(tf.random.normal([2], dtype=tf.bfloat16), [2])
arg_5 = tf.reshape(tf.random.normal([6214], dtype=tf.bfloat16), [478, 13])
arg_6 = tf.reshape(tf.random.uniform([14], minval=5, maxval=30, dtype=tf.int64), [14])
arg_7 = tf.reshape(tf.random.uniform([14], minval=5, maxval=30, dtype=tf.int64), [14])

args = (arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')
# Conversion: torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```