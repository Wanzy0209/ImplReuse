```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True is PyTorch specific, ignored in TF

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(974450504)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0 # size=(17, 30, 17, 3), dtype=bool
    # Conversion: torch.chunk -> tf.split
    var_node_2 = tf.split(var_node_3, 3, axis=3)[0] # size=(17, 30, 17, 1), dtype=bool
    # Conversion: torch.full -> tf.fill + cast
    var_node_5 = tf.cast(tf.fill([17], 3), tf.int64) # size=(17,), dtype=int64
    var_node_6 = arg_1 # size=(15,), dtype=int64
    
    # Conversion: .size(0) -> tf.shape(...)[0]
    _input_size_var_node_4 = tf.shape(var_node_5)[0]
    # Conversion: torch.randint -> tf.random.uniform
    _index_var_node_4 = tf.random.uniform([15], minval=0, maxval=_input_size_var_node_4, dtype=tf.int64)
    # Conversion: torch.gather -> tf.gather
    var_node_4 = tf.gather(var_node_5, _index_var_node_4, axis=0) # size=(15,), dtype=int64
    
    _input_size_var_node_1 = tf.shape(var_node_2)[0]
    _index_var_node_1 = tf.random.uniform([15], minval=0, maxval=_input_size_var_node_1, dtype=tf.int64)
    # Conversion: torch.index_select -> tf.gather
    var_node_1 = tf.gather(var_node_2, _index_var_node_1, axis=0) # size=(15, 30, 17, 1), dtype=bool
    # Conversion: torch.squeeze -> tf.squeeze
    var_node_0 = tf.squeeze(var_node_1) # size=(15, 30, 17), dtype=bool
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # Conversion: is_complex check
    if result.dtype.is_complex:
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# Conversion: torch.tensor(..., requires_grad=True) -> tf.Variable
sentinel = tf.Variable(1.0, dtype=tf.float32)

# Conversion: torch.as_strided with contiguous strides -> direct tensor creation
# The strides (1530, 51, 3, 1) correspond to a standard contiguous layout for shape (17, 30, 17, 3)
arg_0 = tf.cast(tf.random.uniform((17, 30, 17, 3), minval=0, maxval=2, dtype=tf.int32), tf.bool)

# Conversion: torch.as_strided for 1D tensor -> direct tensor creation
arg_1 = tf.random.uniform((15,), minval=5, maxval=30, dtype=tf.int64)

args = (arg_0, arg_1, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# Conversion: torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```