```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# No direct equivalent in TensorFlow, as graph tracing handles scalars naturally.

# torch.manual_seed(740242120)
tf.random.set_seed(740242120)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, sentinel):
    # var_node_4 = torch.full((7, 9), 0.6768777370452881, dtype=torch.float32)
    var_node_4 = tf.cast(tf.fill((7, 9), 0.6768777370452881), tf.float32) # size=(7, 9), dtype=float32

    var_node_5 = arg_0 # size=(9, 8), dtype=float32
    # var_node_3 = torch.matmul(var_node_4.to(torch.float32), var_node_5.to(torch.float32))
    var_node_3 = tf.matmul(var_node_4, var_node_5) # size=(7, 8), dtype=float32

    # var_node_7 = torch.full((8, 10), 0.8236594200134277, dtype=torch.float32)
    var_node_7 = tf.cast(tf.fill((8, 10), 0.8236594200134277), tf.float32) # size=(8, 10), dtype=float32

    # var_node_8 = torch.full((10, 2), 0.022914690896868706, dtype=torch.float32)
    var_node_8 = tf.cast(tf.fill((10, 2), 0.022914690896868706), tf.float32) # size=(10, 2), dtype=float32

    # var_node_6 = torch.matmul(var_node_7.to(torch.float32), var_node_8.to(torch.float32))
    var_node_6 = tf.matmul(var_node_7, var_node_8) # size=(8, 2), dtype=float32

    # var_node_2 = torch.matmul(var_node_3.to(torch.float32), var_node_6.to(torch.float32))
    var_node_2 = tf.matmul(var_node_3, var_node_6) # size=(7, 2), dtype=float32

    var_node_11 = arg_1 # size=(2, 10), dtype=float32
    var_node_12 = arg_2 # size=(10, 13), dtype=float32
    # var_node_10 = torch.matmul(var_node_11.to(torch.float32), var_node_12.to(torch.float32))
    var_node_10 = tf.matmul(var_node_11, var_node_12) # size=(2, 13), dtype=float32

    # var_node_14 = torch.full((13,), 0.06329345703125, dtype=torch.float16)
    var_node_14 = tf.cast(tf.fill((13,), 0.06329345703125), tf.float16) # size=(13,), dtype=float16

    # var_node_15 = torch.full((13,), 0.05013646185398102, dtype=torch.float32)
    var_node_15 = tf.cast(tf.fill((13,), 0.05013646185398102), tf.float32) # size=(13,), dtype=float32

    # var_node_13 = torch.sub(var_node_14, var_node_15)
    # Note: TF promotes types during subtraction. float16 - float32 -> float32.
    var_node_13 = tf.subtract(var_node_14, var_node_15) # size=(13,), dtype=float32

    # var_node_9 = torch.nn.functional.layer_norm(var_node_10.to(torch.float32), (13,), weight=var_node_13.to(torch.float32))
    # PyTorch layer_norm with weight maps to TF layer_normalization with scale.
    # normalized_shape=(13,) implies axis=-1.
    var_node_9 = tf.nn.layer_normalization(var_node_10, scale=var_node_13, offset=None, axis=-1) # size=(2, 13), dtype=float32

    # var_node_1 = torch.matmul(var_node_3.to(torch.float32), var_node_9.to(torch.float32))
    var_node_1 = tf.matmul(var_node_3, var_node_9) # size=(7, 13), dtype=float32

    # var_node_19 = torch.full((13, 13), 0.5703125, dtype=torch.float16)
    var_node_19 = tf.cast(tf.fill((13, 13), 0.5703125), tf.float16) # size=(13, 13), dtype=float16

    # var_node_20 = torch.full((13, 14), -0.051055908203125, dtype=torch.float16)
    var_node_20 = tf.cast(tf.fill((13, 14), -0.051055908203125), tf.float16) # size=(13, 14), dtype=float16

    # var_node_18 = torch.matmul(var_node_19.to(torch.float16), var_node_20.to(torch.float16))
    var_node_18 = tf.matmul(var_node_19, var_node_20) # size=(13, 14), dtype=float16

    var_node_21 = arg_3 # size=(14, 14), dtype=float16
    # var_node_17 = torch.matmul(var_node_18.to(torch.float16), var_node_21.to(torch.float16))
    var_node_17 = tf.matmul(var_node_18, var_node_21) # size=(13, 14), dtype=float16

    var_node_24 = arg_4 # size=(13, 12), dtype=float32
    var_node_25 = arg_5 # size=(12, 11), dtype=float32
    # var_node_23 = torch.matmul(var_node_24.to(torch.float32), var_node_25.to(torch.float32))
    var_node_23 = tf.matmul(var_node_24, var_node_25) # size=(13, 11), dtype=float32

    # var_node_27 = torch.full((11, 15), 1.5650168657302856, dtype=torch.float32)
    var_node_27 = tf.cast(tf.fill((11, 15), 1.5650168657302856), tf.float32) # size=(11, 15), dtype=float32

    # var_node_28 = torch.full((15, 14), 0.0035568897146731615, dtype=torch.float32)
    var_node_28 = tf.cast(tf.fill((15, 14), 0.0035568897146731615), tf.float32) # size=(15, 14), dtype=float32

    # var_node_26 = torch.matmul(var_node_27.to(torch.float32), var_node_28.to(torch.float32))
    var_node_26 = tf.matmul(var_node_27, var_node_28) # size=(11, 14), dtype=float32

    # var_node_22 = torch.matmul(var_node_23.to(torch.float32), var_node_26.to(torch.float32))
    var_node_22 = tf.matmul(var_node_23, var_node_26) # size=(13, 14), dtype=float32

    # var_node_16 = torch.add(var_node_17, var_node_22)
    # Note: TF promotes types during addition. float16 + float32 -> float32.
    var_node_16 = tf.add(var_node_17, var_node_22) # size=(13, 14), dtype=float32

    # var_node_0 = torch.matmul(var_node_1.to(torch.float32), var_node_16.to(torch.float32))
    var_node_0 = tf.matmul(var_node_1, var_node_16) # size=(7, 14), dtype=float32

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # if result.is_complex(): result = result.real
    # TF equivalent: tf.math.real works on real tensors too
    result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# TF doesn't use requires_grad, but we keep the tensor for the multiplication logic
sentinel = tf.constant(1.0)

# arg_0 = torch.as_strided(torch.randn(72).to(torch.float32), (9, 8), (8, 1))
# Translation: tf.random.normal with the target shape
arg_0 = tf.random.normal((9, 8), dtype=tf.float32)
arg_1 = tf.random.normal((2, 10), dtype=tf.float32)
arg_2 = tf.random.normal((10, 13), dtype=tf.float32)
arg_3 = tf.random.normal((14, 14), dtype=tf.float16)
arg_4 = tf.random.normal((13, 12), dtype=tf.float32)
arg_5 = tf.random.normal((12, 11), dtype=tf.float32)

args = (arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, sentinel)

# Eager execution
result_original = fuzzed_program(*args)
print('✅ eager success')

# Compiled execution (tf.function is the TF equivalent of torch.compile for graph mode)
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```