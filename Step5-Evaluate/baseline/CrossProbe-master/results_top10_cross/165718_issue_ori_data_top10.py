```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# No direct equivalent in TensorFlow, compilation behavior is handled by tf.function

tf.random.set_seed(740242120)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, sentinel):
    # Conversion: torch.full -> tf.constant with shape and dtype
    var_node_4 = tf.constant(0.6768777370452881, shape=(7, 9), dtype=tf.float32) # size=(7, 9), stride=(9, 1), dtype=float32, device=cuda
    var_node_5 = arg_0 # size=(9, 8), stride=(8, 1), dtype=float32, device=cuda
    # Conversion: torch.matmul -> tf.matmul, .to() -> tf.cast
    var_node_3 = tf.matmul(tf.cast(var_node_4, tf.float32), tf.cast(var_node_5, tf.float32)) # size=(7, 8), stride=(8, 1), dtype=float32, device=cuda
    
    var_node_7 = tf.constant(0.8236594200134277, shape=(8, 10), dtype=tf.float32) # size=(8, 10), stride=(10, 1), dtype=float32, device=cuda
    var_node_8 = tf.constant(0.022914690896868706, shape=(10, 2), dtype=tf.float32) # size=(10, 2), stride=(2, 1), dtype=float32, device=cuda
    var_node_6 = tf.matmul(tf.cast(var_node_7, tf.float32), tf.cast(var_node_8, tf.float32)) # size=(8, 2), stride=(2, 1), dtype=float32, device=cuda
    var_node_2 = tf.matmul(tf.cast(var_node_3, tf.float32), tf.cast(var_node_6, tf.float32)) # size=(7, 2), stride=(2, 1), dtype=float32, device=cuda
    
    var_node_11 = arg_1 # size=(2, 10), stride=(10, 1), dtype=float32, device=cuda
    var_node_12 = arg_2 # size=(10, 13), stride=(13, 1), dtype=float32, device=cuda
    var_node_10 = tf.matmul(tf.cast(var_node_11, tf.float32), tf.cast(var_node_12, tf.float32)) # size=(2, 13), stride=(13, 1), dtype=float32, device=cuda
    
    var_node_14 = tf.constant(0.06329345703125, shape=(13,), dtype=tf.float16) # size=(13,), stride=(1,), dtype=float16, device=cuda
    var_node_15 = tf.constant(0.05013646185398102, shape=(13,), dtype=tf.float32) # size=(13,), stride=(1,), dtype=float32, device=cuda
    # Conversion: torch.sub -> tf.subtract
    var_node_13 = tf.subtract(var_node_14, var_node_15) # size=(13,), stride=(1,), dtype=float32, device=cuda
    
    # Conversion: torch.nn.functional.layer_norm -> tf.nn.layer_normalization
    # PyTorch 'weight' maps to TensorFlow 'scale'. No bias provided, so offset is None.
    var_node_9 = tf.nn.layer_normalization(tf.cast(var_node_10, tf.float32), (13,), scale=tf.cast(var_node_13, tf.float32), offset=None) # size=(2, 13), stride=(13, 1), dtype=float32, device=cuda
    
    var_node_1 = tf.matmul(tf.cast(var_node_2, tf.float32), tf.cast(var_node_9, tf.float32)) # size=(7, 13), stride=(13, 1), dtype=float32, device=cuda
    
    var_node_19 = tf.constant(0.5703125, shape=(13, 13), dtype=tf.float16) # size=(13, 13), stride=(13, 1), dtype=float16, device=cuda
    var_node_20 = tf.constant(-0.051055908203125, shape=(13, 14), dtype=tf.float16) # size=(13, 14), stride=(14, 1), dtype=float16, device=cuda
    var_node_18 = tf.matmul(tf.cast(var_node_19, tf.float16), tf.cast(var_node_20, tf.float16)) # size=(13, 14), stride=(14, 1), dtype=float16, device=cuda
    var_node_21 = arg_3 # size=(14, 14), stride=(14, 1), dtype=float16, device=cuda
    var_node_17 = tf.matmul(tf.cast(var_node_18, tf.float16), tf.cast(var_node_21, tf.float16)) # size=(13, 14), stride=(14, 1), dtype=float16, device=cuda
    
    var_node_24 = arg_4 # size=(13, 12), stride=(12, 1), dtype=float32, device=cuda
    var_node_25 = arg_5 # size=(12, 11), stride=(11, 1), dtype=float32, device=cuda
    var_node_23 = tf.matmul(tf.cast(var_node_24, tf.float32), tf.cast(var_node_25, tf.float32)) # size=(13, 11), stride=(11, 1), dtype=float32, device=cuda
    
    var_node_27 = tf.constant(1.5650168657302856, shape=(11, 15), dtype=tf.float32) # size=(11, 15), stride=(15, 1), dtype=float32, device=cuda
    var_node_28 = tf.constant(0.0035568897146731615, shape=(15, 14), dtype=tf.float32) # size=(15, 14), stride=(14, 1), dtype=float32, device=cuda
    var_node_26 = tf.matmul(tf.cast(var_node_27, tf.float32), tf.cast(var_node_28, tf.float32)) # size=(11, 14), stride=(14, 1), dtype=float32, device=cuda
    var_node_22 = tf.matmul(tf.cast(var_node_23, tf.float32), tf.cast(var_node_26, tf.float32)) # size=(13, 14), stride=(14, 1), dtype=float32, device=cuda
    
    # Conversion: torch.add -> tf.add
    var_node_16 = tf.add(var_node_17, var_node_22) # size=(13, 14), stride=(14, 1), dtype=float32, device=cuda
    
    var_node_0 = tf.matmul(tf.cast(var_node_1, tf.float32), tf.cast(var_node_16, tf.float32)) # size=(7, 14), stride=(1, 1), dtype=float32, device=cuda
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # Conversion: result.is_complex() check and .real -> tf.math.real (handles both real and complex)
    result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# Conversion: torch.tensor(..., requires_grad=True) -> tf.Variable
sentinel = tf.Variable(1.0, dtype=tf.float32)

# Conversion: torch.as_strided with standard strides -> tf.reshape
# The strides (dim[1], 1) indicate standard row-major layout, which is the default for tf.reshape
arg_0 = tf.reshape(tf.random.normal((72,), dtype=tf.float32), (9, 8))
arg_1 = tf.reshape(tf.random.normal((20,), dtype=tf.float32), (2, 10))
arg_2 = tf.reshape(tf.random.normal((130,), dtype=tf.float32), (10, 13))
arg_3 = tf.reshape(tf.random.normal((196,), dtype=tf.float16), (14, 14))
arg_4 = tf.reshape(tf.random.normal((156,), dtype=tf.float32), (13, 12))
arg_5 = tf.reshape(tf.random.normal((132,), dtype=tf.float32), (12, 11))

args = (arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# Conversion: torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```