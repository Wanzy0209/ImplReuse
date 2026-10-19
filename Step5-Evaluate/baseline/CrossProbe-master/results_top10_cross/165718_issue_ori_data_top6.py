```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# Note: TensorFlow's tf.function handles graph compilation differently; 
# this specific PyTorch config has no direct equivalent.

tf.random.set_seed(740242120)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, sentinel):
    # Conversion: torch.full -> tf.constant
    var_node_4 = tf.constant(0.6768777370452881, shape=(7, 9), dtype=tf.float32) # size=(7, 9), dtype=float32
    var_node_5 = arg_0 # size=(9, 8), dtype=float32
    
    # Conversion: torch.matmul -> tf.matmul
    # Conversion: .to(torch.float32) -> tf.cast(..., tf.float32)
    var_node_3 = tf.matmul(tf.cast(var_node_4, tf.float32), tf.cast(var_node_5, tf.float32)) # size=(7, 8), dtype=float32
    
    var_node_7 = tf.constant(0.8236594200134277, shape=(8, 10), dtype=tf.float32) # size=(8, 10), dtype=float32
    var_node_8 = tf.constant(0.022914690896868706, shape=(10, 2), dtype=tf.float32) # size=(10, 2), dtype=float32
    var_node_6 = tf.matmul(tf.cast(var_node_7, tf.float32), tf.cast(var_node_8, tf.float32)) # size=(8, 2), dtype=float32
    var_node_2 = tf.matmul(tf.cast(var_node_3, tf.float32), tf.cast(var_node_6, tf.float32)) # size=(7, 2), dtype=float32
    
    var_node_11 = arg_1 # size=(2, 10), dtype=float32
    var_node_12 = arg_2 # size=(10, 13), dtype=float32
    var_node_10 = tf.matmul(tf.cast(var_node_11, tf.float32), tf.cast(var_node_12, tf.float32)) # size=(2, 13), dtype=float32
    
    var_node_14 = tf.constant(0.06329345703125, shape=(13,), dtype=tf.float16) # size=(13,), dtype=float16
    var_node_15 = tf.constant(0.05013646185398102, shape=(13,), dtype=tf.float32) # size=(13,), dtype=float32
    
    # Conversion: torch.sub -> tf.subtract
    # Note: Explicit cast to float32 to match PyTorch promotion behavior
    var_node_13 = tf.subtract(tf.cast(var_node_14, tf.float32), var_node_15) # size=(13,), dtype=float32
    
    # Conversion: torch.nn.functional.layer_norm -> tf.nn.layer_normalization
    # PyTorch normalized_shape (13,) on input (2, 13) implies normalizing over the last axis.
    # PyTorch weight maps to TF scale.
    var_node_9 = tf.nn.layer_normalization(
        tf.cast(var_node_10, tf.float32), 
        scale=tf.cast(var_node_13, tf.float32), 
        offset=None, 
        axis=-1, 
        epsilon=