```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# TensorFlow handles scalar outputs in graphs by default, no specific config needed.

tf.random.set_seed(740242120)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, sentinel):
    # var_node_4 = torch.full((7, 9), 0.6768777370452881, dtype=torch.float32)
    # Conversion: torch.full -> tf.fill + tf.cast
    var_node_4 = tf.cast(tf.fill((7, 9), 0.6768777370452881), tf.float32) # size=(7, 9), dtype=float32

    var_node_5 = arg_0 # size=(9, 8), dtype=float32

    # var_node_3 = torch.matmul(var_node_4.to(torch.float32), var_node_5.to(torch.float32))
    # Conversion: torch.matmul -> tf.matmul
    var_node_3 = tf.matmul(var_node_4, tf.cast(var_node_5, tf.float32)) # size=(7, 8), dtype=float32

    # var_node_7 = torch.full((8, 10), 0.8236594200134277, dtype=torch.float32)
    var_node_7 = tf.cast(tf.fill((8, 10), 0.8236594200134277), tf.float32) # size=(8, 10), dtype=float32

    # var_node_8 = torch.full((10, 2), 0.022914690896868706, dtype=torch.float32)
    var_node_8 = tf.cast(tf.fill((10, 2), 0.022914690896868706), tf.float32) # size=(10, 2), dtype=float32

    var_node_6 = tf.matmul(var_node_7, var_node_8) # size=(8, 2), dtype=float32

    var_node_2 = tf.matmul(var_node_3, var_node_6) # size=(7, 2), dtype=float32

    var_node_11 = arg_1 # size=(2, 10), dtype=float32
    var_node_12 = arg_2 # size=(10, 13), dtype=float32
    var_node_10 = tf.matmul(tf.cast(var_node_11, tf.float32), tf.cast(var_node_12, tf.float32)) # size=(2, 13), dtype=float32

    # var_node_14 = torch.full((13,), 0.06329345703125, dtype=torch.float16)
    var_node_14 = tf.cast(tf.fill((13,), 0.06329345703125), tf.float16) # size=(13,), dtype=float16

    # var_node_15 = torch.full((13,), 0.05013646185398102, dtype=torch.float32)
    var_node_15 = tf.cast(tf.fill((13,), 0.05013646185398102), tf.float32) # size=(13,), dtype=float32

    # var_node_13 = torch.sub(var_node_14, var_node_15)
    # Conversion: torch.sub -> tf.subtract
    # Note: PyTorch promotes float16 - float32 to float32. TF does the same.
    var_node_13 = tf.subtract(var_node_14, var_node_15) # size=(13,), dtype=float32

    # var_node_9 = torch.nn.functional.layer_norm(var_node_10.to(torch.float32), (13,), weight=var_node_13.to(torch.float32))
    # Conversion: torch.nn.functional.layer_norm -> tf.nn.layer_normalization
    # PyTorch 'weight' maps to TF 'scale'.
    var_node_9 = tf.nn.layer_normalization(var_node_10, (13,), scale=var_node_13, offset=None) # size=(2, 13), dtype=float32

    var_node_1 = tf.matmul(var_node_2, var_node_9) # size=(7, 13), dtype=float32

    # var_node_19 = torch.full((13, 13), 0.5703125, dtype=torch.float16)
    var_node_19 = tf.cast(tf.fill((13, 13), 0.5703125), tf.float16) # size=(13, 13), dtype=float16

    # var_node_20 = torch.full((13, 14), -0.051055908203125, dtype=torch.float16)
    var_node_20 = tf.cast(tf.fill((13, 14), -0.051055908203125), tf.float16) # size=(13, 14), dtype=float16

    var_node_18 = tf.matmul(var_node_19, var_node_20) # size=(13, 14), dtype=float16

    var_node_21 = arg_3 # size=(14, 14), dtype=float16
    var_node_17 = tf.matmul(var_node_18, tf.cast(var_node_21, tf.float16)) # size=(13, 14), dtype=float16

    var_node_24 = arg_4 # size=(13, 12), dtype=float32
    var_node_25 = arg_5 # size=(12, 11), dtype=float32
    var_node_23 = tf.matmul(tf.cast(var_node_24, tf.float32), tf.cast(var_node_25, tf.float32)) # size=(13, 11), dtype=float32

    # var_node_27 = torch.full((11, 15), 1.5650168657302856, dtype=torch.float32)
    var_node_27 = tf.cast(tf.fill((11, 15), 1.5650168657302856), tf.float32) # size=(11, 15), dtype=float32

    # var_node_28 = torch.full((15, 14), 0.0035568897146731615, dtype=torch.float32)
    var_node_28 = tf.cast(tf.fill((15, 14), 0.0035568897146731615), tf.float32) # size=(15, 14), dtype=float32

    var_node_26 = tf.matmul(var_node_27, var_node_28) # size=(11, 14), dtype=float32

    var_node_22 = tf.matmul(var_node_23, var_node_26) # size=(13, 14), dtype=float32

    # var_node_16 = torch.add(var_node_17, var_node_22)
    # Conversion: torch.add -> tf.add
    # Note: PyTorch promotes float16 + float32 to float32. TF does the same.
    var_node_16 = tf.add(var_node_17, var_node_22) # size=(13, 14), dtype=float32

    var_node_0 = tf.matmul(var_node_1, var