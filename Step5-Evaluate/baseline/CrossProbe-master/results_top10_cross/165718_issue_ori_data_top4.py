```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# No direct equivalent in TensorFlow, graph tracing handles scalar outputs by default.

# torch.manual_seed(740242120)
tf.random.set_seed(740242120)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, sentinel):
    # var_node_4 = torch.full((7, 9), 0.6768777370452881, dtype=torch.float32)
    var_node_4 = tf.fill((7, 9), 0.6768777370452881) # size=(7, 9), dtype=float32

    var_node_5 = arg_0 # size=(9, 8), dtype=float32

    # var_node_3 = torch.matmul(var_node_4.to(torch.float32), var_node_5.to(torch.float32))
    var_node_3 = tf.matmul(tf.cast(var_node_4, tf.float32), tf.cast(var_node_5, tf.float32)) # size=(7, 8), dtype=float32

    # var_node_7 = torch.full((8, 10), 0.8236594200134277, dtype=torch.float32)
    var_node_7 = tf.fill((8, 10), 0.8236594200134277) # size=(8, 10), dtype=float32

    # var_node_8 = torch.full((10, 2), 0.022914690896868706, dtype=torch.float32)
    var_node_8 = tf.fill((10, 2), 0.022914690896868706) # size=(10, 2), dtype=float32

    # var_node_6 = torch.matmul(var_node_7.to(torch.float32), var_node_8.to(torch.float32))
    var_node_6 = tf.matmul(tf.cast(var_node_7, tf.float32), tf.cast(var_node_8, tf.float32)) # size=(8, 2), dtype=float32

    # var_node_2 = torch.matmul(var_node_3.to(torch.float32), var_node_6.to(torch.float32))
    var_node_2 = tf.matmul(tf.cast(var_node_3, tf.float32), tf.cast(var_node_6, tf.float32)) # size=(7, 2), dtype=float32

    var_node_11 = arg_1 # size=(2, 10), dtype=float32
    var_node_12 = arg_2 # size=(10, 13), dtype=float32

    # var_node_10 = torch.matmul(var_node_11.to(torch.float32), var_node_12.to(torch.float32))
    var_node_10 = tf.matmul(tf.cast(var_node_11, tf.float32), tf.cast(var_node_12, tf.float32)) # size=(2, 13), dtype=float32

    # var_node_14 = torch.full((13,), 0.06329345703125, dtype=torch.float16)
    var_node_14 = tf.cast(tf.fill((13,), 0.06329345703125), tf.float16) # size=(13,), dtype=float16

    # var_node_15 = torch.full((13,), 0.05013646185398102, dtype=torch.float32)
    var_node_15 = tf.fill((13,), 0.05013646185398102) # size=(13,), dtype=float32

    # var_node_13 = torch.sub(var_node_14, var_node_15)
    var_node_13 = tf.subtract(var_node_14, var_node_15) # size=(13,), dtype=float32 (promotion)

    # var_node_9 = torch.nn.functional.layer_norm(var_node_10.to(torch.float32), (13,), weight=var_node_13.to(torch.float32))
    # PyTorch layer_norm normalizes over the last N dimensions. Here (13,) implies last dim.
    # TF layer_norm normalizes over the specified axes.
    # PyTorch default eps is 1e-5, TF default is 0.001, so we set epsilon explicitly.
    var_node_9 = tf.nn.layer_normalization(
        tf.cast(var_node_10, tf.float32),
        scale=tf.cast(var_node_13, tf.float32),
        offset=None, # No bias provided
        axis=-1,
        epsilon=1e-05
    ) # size=(2, 13), dtype=float32

    # var_node_1 = torch.matmul(var_node_2.to(torch.float32), var_node_9.to(torch.float32))
    var_node_1 = tf.matmul(tf.cast(var_node_2, tf.float32), tf.cast(var_node_9, tf.float32)) # size=(7, 13), dtype=float32

    # var_node_19 = torch.full((13, 13), 0.5703125, dtype=torch.float16)
    var_node_19 = tf.cast(tf.fill((13, 13), 0.5703125), tf.float16) # size=(13, 13), dtype=float16

    # var_node_20 = torch.full((13, 14), -0.051055908203125, dtype=torch.float16)
    var_node_20 = tf.cast(tf.fill((13, 14), -0.051055908203125), tf.float16) # size=(13, 14), dtype=float16

    # var_node_18 = torch.matmul(var_node_19.to(torch.float16), var_node_20.to(torch.float16))
    var_node_18 = tf.matmul(var_node_19, var_node_20) # size=(13, 14), dtype=float16

    var_node_