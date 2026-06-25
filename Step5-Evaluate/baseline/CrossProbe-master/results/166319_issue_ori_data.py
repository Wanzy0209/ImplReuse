```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# Note: TensorFlow's tf.function handles scalar outputs automatically.

tf.random.set_seed(1215252001)

def fuzzed_program(arg_0, arg_1, arg_2, sentinel):
    var_node_4 = arg_0 # size=(9, 1, 15, 4), stride=(60, 60, 0, 1), dtype=int32
    var_node_3 = tf.squeeze(var_node_4) # size=(9, 15, 4)
    var_node_2 = tf.split(var_node_3, 4, axis=2)[0] # size=(9, 15, 1)
    var_node_1 = tf.squeeze(var_node_2) # size=(9, 15)
    
    var_node_7 = arg_1 # size=(20, 15), dtype=int64
    var_node_8 = arg_2 # size=(18, 15), dtype=int64
    
    _input_size_var_node_6 = tf.shape(var_node_7)[0]
    _index_var_node_6 = tf.random.uniform((18, 15), minval=0, maxval=_input_size_var_node_6, dtype=tf.int64)
    
    # Conversion: torch.gather(input, dim, index) -> tf.gather_nd
    # For dim=0, we construct coordinates (index[i, j], j)
    row_indices = _index_var_node_6
    col_indices = tf.tile(tf.range(15, dtype=tf.int64)[tf.newaxis, :], [18, 1])
    gather_indices = tf.stack([row_indices, col_indices], axis=-1)
    var_node_6 = tf.gather_nd(var_node_7, gather_indices) # size=(18, 15), dtype=int64
    
    var_node_5 = tf.split(var_node_6, 2, axis=0)[0] # size=(9, 15)
    
    var_node_0 = tf.multiply(var_node_1, var_node_5) # size=(9, 15), dtype=int64
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if tf.dtypes.is_complex(result.dtype):
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
sentinel = tf.constant(1.0)

# Conversion: torch.as_strided with stride 0 implies broadcasting.
# We construct the tensor by gathering specific elements and broadcasting.
# Base tensor generation
base_0 = tf.random.uniform((484,), minval=5, maxval=30, dtype=tf.int32)
# Indices for strided access: i*60 + k for i in 0..8, k in 0..3
i_vals = tf.range(9)[:, tf.newaxis] * 60
k_vals = tf.range(4)[tf.newaxis, :]
flat_indices = tf.reshape(i_vals + k_vals, [-1])
gathered_0 = tf.gather(base_0, flat_indices)
arg_0 = tf.reshape(gathered_0, (9, 1, 1, 4))
arg_0 = tf.broadcast_to(arg_0, (9, 1, 15, 4))

# Conversion: torch.as_strided with contiguous strides is equivalent to reshape
base_1 = tf.random.uniform((300,), minval=5, maxval=30, dtype=tf.int64)
arg_1 = tf.reshape(base_1, (20, 15))

base_2 = tf.random.uniform((270,), minval=5, maxval=30, dtype=tf.int64)
arg_2 = tf.reshape(base_2, (18, 15))

args = (arg_0, arg_1, arg_2, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# Conversion: torch.compile -> tf.function with jit_compile=True
compiled_program = tf.function(fuzzed_program, jit_compile=True)
result_compiled = compiled_program(*args)
print('✅ compile success')
```