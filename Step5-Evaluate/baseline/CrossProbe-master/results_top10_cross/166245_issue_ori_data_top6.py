```python
import tensorflow as tf

# Set random seed for reproducibility
tf.random.set_seed(751735337)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel):
    var_node_4 = arg_0 # size=(15, 108, 4), dtype=int16
    # Conversion: torch.chunk -> tf.split
    var_node_3 = tf.split(var_node_4, 4, axis=1)[0] # size=(15, 27, 4), dtype=int16
    var_node_2 = tf.split(var_node_3, 4, axis=2)[0] # size=(15, 27, 1), dtype=int16
    # Conversion: torch.squeeze -> tf.squeeze
    var_node_1 = tf.squeeze(var_node_2) # size=(15, 27), dtype=int16
    
    # Conversion: torch.full -> tf.fill + cast
    var_node_8 = tf.cast(tf.fill((13, 27), 3), tf.int16) # size=(13, 27), dtype=int16
    var_node_9 = arg_1 # size=(11,), dtype=int64
    
    # Conversion: var.size(0) -> tf.shape(var)[0]
    _input_size_var_node_7 = tf.shape(var_node_8)[0]
    # Conversion: torch.randint -> tf.random.uniform (cast to int64 for indices)
    _index_var_node_7 = tf.cast(tf.random.uniform((11,), minval=0, maxval=_input_size_var_node_7, dtype=tf.int32), tf.int64)
    # Conversion: torch.index_select -> tf.gather
    var_node_7 = tf.gather(var_node_8, _index_var_node_7, axis=0) # size=(11, 27), dtype=int16
    
    # Conversion: torch.clamp -> tf.clip_by_value
    var_node_6 = tf.clip_by_value(var_node_7, clip_value_min=-1.0, clip_value_max=1.0) # size=(11, 27), dtype=int16
    
    var_node_12 = arg_2 # size=(3, 27), dtype=int16
    var_node_11 = tf.clip_by_value(var_node_12, clip_value_min=-1.0, clip_value_max=1.0) # size=(3, 27), dtype=int16
    
    var_node_13 = tf.cast(tf.fill((1,), 3), tf.int64) # size=(1,), dtype=int64
    
    _input_size_var_node_10 = tf.shape(var_node_11)[0]
    _index_var_node_10 = tf.cast(tf.random.uniform((1,), minval=0, maxval=_input_size_var_node_10, dtype=tf.int32), tf.int64)
    var_node_10 = tf.gather(var_node_11, _index_var_node_10, axis=0) # size=(1, 27), dtype=int16
    
    var_node_16 = arg_3 # size=(1, 27), dtype=int16
    var_node_17 = arg_4 # size=(1, 27), dtype=int16
    var_node_18 = arg_5 # size=(1, 27), dtype=int16
    
    # Conversion: torch.cat -> tf.concat
    var_node_15 = tf.concat([var_node_16, var_node_17, var_node_18], axis=0) # size=(3, 27), dtype=int16
    
    var_node_20 = arg_6 # size=(1,), dtype=int64
    # Conversion: torch.clamp(min=None) -> tf.minimum
    var_node_19 = tf.minimum(var_node_20, tf.cast(1.0, tf.int64)) # size=(1,), dtype=int64
    
    _input_size_var_node_14 = tf.shape(var_node_15)[0]
    _index_var_node_14 = tf.cast(tf.random.uniform((1,), minval=0, maxval=_input_size_var_node_14, dtype=tf.int32), tf.int64)
    var_node_14 = tf.gather(var_node_15, _index_var_node_14, axis=0) # size=(1, 27), dtype=int16
    
    var_node_22 = tf.cast(tf.fill((4, 27), 3), tf.int16) # size=(4, 27), dtype=int16
    var_node_24 = tf.cast(tf.fill((4,), 3), tf.int64) # size=(4,), dtype=int64
    var_node_25 = tf.cast(tf.fill((2,), 3), dtype=tf.int64) # size=(2,), dtype=int64
    
    _input_size_var_node_23 = tf.shape(var_node_24)[0]
    _index_var_node_23 = tf.cast(tf.random.uniform((2,), minval=0, maxval=_input_size_var_node_23, dtype=tf.int32), tf.int64)
    # Conversion: torch.gather