```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True # No direct equivalent in TF
# torch._dynamo.config.capture_dynamic_output_shape_ops = True # No direct equivalent in TF

tf.random.set_seed(70609)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, sentinel):
    var_node_4 = tf.fill((14,), tf.cast(1.2255859375, tf.float16)) # size=(14,), dtype=float16
    var_node_6 = arg_0 # size=(14, 6), dtype=float16
    var_node_9 = tf.fill((6, 13), tf.cast(1.3154296875, tf.float16)) # size=(6, 13), dtype=float16
    var_node_10 = arg_1 # size=(13, 1), dtype=float16
    var_node_8 = tf.matmul(tf.cast(var_node_9, tf.float16), tf.cast(var_node_10, tf.float16)) # size=(6, 1), dtype=float16
    var_node_12 = arg_2 # size=(1, 10), dtype=float16
    var_node_13 = tf.fill((10, 416), tf.cast(0.1331787109375, tf.float16)) # size=(10, 416), dtype=float16
    var_node_11 = tf.matmul(tf.cast(var_node_12, tf.float16), tf.cast(var_node_13, tf.float16)) # size=(1, 416), dtype=float16
    var_node_7 = tf.matmul(tf.cast(var_node_8, tf.float16), tf.cast(var_node_11, tf.float16)) # size=(6, 416), dtype=float16
    var_node_5 = tf.matmul(tf.cast(var_node_6, tf.float16), tf.cast(var_node_7, tf.float16)) # size=(14, 416), dtype=float16
    var_node_3 = tf.matmul(tf.cast(var_node_4, tf.float16), tf.cast(var_node_5, tf.float16)) # size=(416,), dtype=float16
    var_node_2 = tf.reshape(var_node_3, [26, 1, 16]) # size=(26, 1, 16), dtype=float16
    var_node_16 = arg_3 # size=(329, 4), dtype=float16
    var_node_20 = arg_4 # size=(1,), dtype=bool
    var_node_19 = tf.reshape(var_node_20, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) # size=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=bool
    var_node_18 = tf.expand_dims(var_node_19, axis=15) # size=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=bool
    
    # Constructing _x_nz with first 26 elements True
    _x_nz_shape = (26, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    _x_nz_flat = tf.concat([tf.ones([26], dtype=tf.bool), tf.zeros([tf.reduce_prod(_x_nz_shape) - 26], dtype=tf.bool)], axis=0)
    _x_nz = tf.reshape(_x_nz_flat, _x_nz_shape)
    
    var_node_17 = tf.where(_x_nz) # size=(26, 16), dtype=int64
    var_node_15 = tf.gather(var_node_16, tf.clip_by_value(tf.cast(var_node_17, tf.int64), 0, tf.shape(var_node_16)[0] - 1)) # size=(26, 16, 4), dtype=float16
    var_node_25 = arg_5 # size=(26, 4, 8), dtype=float16
    var_node_26 = arg_6 # size=(26, 8, 11), dtype=float16
    var_node_24 = tf.matmul(tf.cast(var_node_25, tf.float16), tf.cast(var_node_26, tf.float16)) # size=(26, 4, 11), dtype=float16
    var_node_28 = tf.fill((26, 11, 16), tf.cast(-0.73291015625, tf.float16)) # size=(26, 11, 16), dtype=float16
    var_node_29 = tf.fill((26, 16, 16), tf.cast(-0.53271484375, tf.float16)) # size=(26, 16, 16), dtype=float16
    var_node_27 = tf.matmul(tf.cast(var_node_28, tf.float16), tf.cast(var_node_29, tf.float16)) # size=(26, 11, 16), dtype=float16
    var_node_23 = tf.matmul(tf.cast(var_node_24, tf.float16), tf.cast(var_node_27, tf.float16)) # size=(26, 4, 16), dtype=float16
    var_node_31 = arg_7 # size=(26, 16, 11), dtype=float16
    var_node_33 = arg_8 # size=(26, 11, 11), dtype=float16
    var_node_34 = tf.fill((26, 11, 6), tf.cast(-0.51318359375, tf.float16)) # size=(26, 11, 6), dtype=float