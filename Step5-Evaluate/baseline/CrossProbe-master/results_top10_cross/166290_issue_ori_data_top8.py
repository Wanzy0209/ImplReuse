```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# No direct equivalent in TensorFlow, handled by tf.function behavior

# torch.manual_seed(974450504)
tf.random.set_seed(974450504)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0 # size=(17, 30, 17, 3), stride=(1530, 51, 3, 1), dtype=bool, device=cuda
    
    # var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
    # Conversion: tf.split splits tensor into num_split parts along axis.
    var_node_2 = tf.split(var_node_3, 3, axis=3)[0] # size=(17, 30, 17, 1), stride=(510, 17, 1, 1), dtype=bool, device=cuda
    
    # var_node_5 = torch.full((17,), 3, dtype=torch.int64)
    # Conversion: tf.fill creates a tensor filled with a scalar value.
    var_node_5 = tf.fill((17,), tf.cast(3, tf.int64)) # size=(17,), stride=(1,), dtype=int64, device=cuda
    
    var_node_6 = arg_1 # size=(15,), stride=(1,), dtype=int64, device=cuda
    
    # _input_size_var_node_4 = var_node_5.size(0)
    # Conversion: tf.shape returns the shape tensor.
    _input_size_var_node_4 = tf.shape(var_node_5)[0]
    
    # _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    # Conversion: tf.random.uniform generates random values. minval is inclusive, maxval is exclusive.
    _index_var_node_4 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_4, dtype=tf.int64)
    
    # var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
    # Conversion: tf.gather gathers slices from params axis according to indices.
    var_node_4 = tf.gather(var_node_5, _index_var_node_4, axis=0) # size=(15,), stride=(1,), dtype=int64, device=cuda
    
    # _input_size_var_node_1 = var_node_2.size(0)
    _input_size_var_node_1 = tf.shape(var_node_2)[0]
    
    # _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    _index_var_node_1 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_1, dtype=tf.int64)
    
    # var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
    # Conversion: tf.gather is the equivalent of index_select.
    var_node_1 = tf.gather(var_node_2, _index_var_node_1, axis=0) # size=(15, 30, 17, 1), stride=(510, 17, 1, 1), dtype=bool, device=cuda
    
    # var_node_0 = torch.squeeze(var_node_1)
    # Conversion: tf.squeeze removes dimensions of size 1.
    var_node_0 = tf.squeeze(var_node_1) # size=(15, 30, 17), stride=(510, 17, 1), dtype=bool, device=cuda
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    
    # if result.is_complex(): result = result.real
    # Conversion: Check if complex and take real part.
    # tf.math.real requires complex input, so we use tf.cond.
    result = tf.cond(tf.math.is_complex(result), lambda: tf.math.real(result), lambda: result)
    
    return result

# Sentinel tensor to ensure gradient computation
# Conversion: In TF, gradients are handled via GradientTape, but for the tensor definition:
sentinel = tf.constant(1.0)

# arg_0 = torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), (17, 30, 17, 3), (1530, 51, 3, 1))
# Conversion: The strides indicate a standard reshape (C-contiguous).
# torch.randint(0, 2, ...) -> tf.random.uniform(..., 0, 2, dtype=tf.int64)
# .bool() -> tf.cast(..., tf.bool)
# .as_strided(...) -> tf.reshape(...)
arg_0 = tf.reshape(tf.cast(tf.random.uniform((26010,), 0, 2, dtype=tf.int64), tf.bool), (17, 30, 17, 3))

# arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,))
# Conversion: Standard reshape.
arg_1 = tf.reshape(tf.random.uniform((15,), 5, 30, dtype=tf.int64), (15,))

args = (arg_0, arg_1, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Conversion: tf.function compiles the TensorFlow graph.
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```