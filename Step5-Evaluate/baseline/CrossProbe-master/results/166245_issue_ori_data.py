```python
import tensorflow as tf

# Conversion: torch._dynamo.config.capture_scalar_outputs = True -> Not applicable in TF
# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(751735337)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel):
    var_node_4 = arg_0 # size=(15, 108, 4), stride=(432, 1, 4), dtype=int16, device=cuda
    # Conversion: torch.chunk -> tf.split
    var_node_3 = tf.split(var_node_4, 4, axis=1)[0] # size=(15, 27, 4), stride=(108, 4, 1), dtype=int16, device=cuda
    var_node_2 = tf.split(var_node_3, 4, axis=2)[0] # size=(15, 27, 1), stride=(27, 1, 1), dtype=int16, device=cuda
    # Conversion: torch.squeeze -> tf.squeeze
    var_node_1 = tf.squeeze(var_node_2) # size=(15, 27), stride=(1, 1), dtype=int16, device=cuda
    # Conversion: torch.full -> tf.fill
    var_node_8 = tf.fill((13, 27), tf.constant(3, dtype=tf.int16)) # size=(13, 27), stride=(27, 1), dtype=int16, device=cuda
    var_node_9 = arg_1 # size=(11,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_7 = tf.shape(var_node_8)[0]
    # Conversion: torch.randint -> tf.random.uniform
    _index_var_node_7 = tf.random.uniform((11,), minval=0, maxval=_input_size_var_node_7, dtype=tf.int64)
    # Conversion: torch.index_select -> tf.gather
    var_node_7 = tf.gather(var_node_8, _index_var_node_7, axis=0) # size=(11, 27), stride=(1, 1), dtype=int16, device=cuda
    # Conversion: torch.clamp -> tf.clip_by_value
    var_node_6 = tf.clip_by_value(var_node_7, -1.0, 1.0) # size=(11, 27), stride=(1, 1), dtype=int16, device=cuda
    var_node_12 = arg_2 # size=(3, 27), stride=(27, 1), dtype=int16, device=cuda
    var_node_11 = tf.clip_by_value(var_node_12, -1.0, 1.0) # size=(3, 27), stride=(27, 1), dtype=int16, device=cuda
    var_node_13 = tf.fill((1,), tf.constant(3, dtype=tf.int64)) # size=(1,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_10 = tf.shape(var_node_11)[0]
    _index_var_node_10 = tf.random.uniform((1,), minval=0, maxval=_input_size_var_node_10, dtype=tf.int64)
    var_node_10 = tf.gather(var_node_11, _index_var_node_10, axis=0) # size=(1, 27), stride=(27, 1), dtype=int16, device=cuda
    var_node_16 = arg_3 # size=(1, 27), stride=(1, 1), dtype=int16, device=cuda
    var_node_17 = arg_4 # size=(1, 27), stride=(27, 1), dtype=int16, device=cuda
    var_node_18 = arg_5 # size=(1, 27), stride=(27, 1), dtype=int16, device=cuda
    # Conversion: torch.cat -> tf.concat
    var_node_15 = tf.concat([var_node_16, var_node_17, var_node_18], axis=0) # size=(3, 27), stride=(27, 1), dtype=int16, device=cuda
    var_node_20 = arg_6 # size=(1,), stride=(1,), dtype=int64, device=cuda
    # Conversion: torch.clamp(min=None) -> tf.clip_by_value with min=-inf
    var_node_19 = tf.clip_by_value(var_node_20, tf.cast(-1e9, tf.int64), 1.0) # size=(1,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_14 = tf.shape(var_node_15)[0]
    _index_var_node_14 = tf.random.uniform((1,), minval=0, maxval=_input_size_var_node_14, dtype=tf.int64)
    var_node_14 = tf.gather(var_node_15, _index_var_node_14, axis=0) # size=(1, 27), stride=(27, 1), dtype=int16, device=cuda
    var_node_22 = tf.fill((4, 27), tf.constant(3, dtype=tf.int16)) # size=(4, 27), stride=(27, 1), dtype=int16, device=cuda
    var_node_24 = tf.fill((4,), tf.constant(3, dtype=tf.int64)) # size=(4,), stride=(1,), dtype=int64, device=cuda
    var_node_25 = tf.fill((2,), tf.constant(3, dtype=tf.int64)) # size=(2,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_23 = tf.shape(var_node_24)[0]
    _index_var_node_23 = tf.random.uniform((2,), minval=0, maxval=_input_size_var_node_23, dtype=tf.int64)
    # Conversion: torch.gather -> tf.gather
    var_node_23 = tf.gather(var_node_24, _index_var_node_23, axis=0) # size=(2,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_21 = tf.shape(var_node_22)[0]
    _index_var_node_21 = tf.random.uniform((2,), minval=0, maxval=_input_size_var_node_21, dtype=tf.int64)
    var_node_21 = tf.gather(var_node_22, _index_var_node_21, axis=0) # size=(2, 27), stride=(1, 27), dtype=int16, device=cuda
    var_node_5 = tf.concat([var_node_6, var_node_10, var_node_14, var_node_21], axis=0) # size=(15, 27), stride=(1, 1), dtype=int16, device=cuda
    # Conversion: torch.div -> tf.math.floordiv (for int types)
    var_node_0 = tf.cast(tf.math.floordiv(var_node_1, var_node_5), tf.int16) # size=(15, 27), stride=(1, 1), dtype=int16, device=cuda
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # Conversion: result.is_complex() -> tf.dtypes.complex check
    if tf.dtypes.complex(result.dtype):
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# Conversion: torch.tensor -> tf.constant
sentinel = tf.constant(1.0)

# Conversion: torch.as_strided + torch.randint -> tf.random.uniform with target shape
# Note: as_strided creates a view, here we approximate by creating the target shape directly
arg_0 = tf.random.uniform((15, 108, 4), minval=5, maxval=30, dtype=tf.int16)
arg_1 = tf.random.uniform((11,), minval=5, maxval=30, dtype=tf.int64)
arg_2 = tf.random.uniform((3, 27), minval=5, maxval=30, dtype=tf.int16)
arg_3 = tf.random.uniform((1, 27), minval=5, maxval=30, dtype=tf.int16)
arg_4 = tf.random.uniform((1, 27), minval=5, maxval=30, dtype=tf.int16)
arg_5 = tf.random.uniform((1, 27), minval=5, maxval=30, dtype=tf.int16)
arg_6 = tf.random.uniform((1,), minval=5, maxval=30, dtype=tf.int64)

args = (arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
# Conversion: torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```