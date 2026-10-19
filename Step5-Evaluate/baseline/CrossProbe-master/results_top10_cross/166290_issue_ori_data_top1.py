```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# No direct equivalent needed in TF 2.x as scalars are handled automatically in graphs.

# torch.manual_seed(974450504)
# Conversion: Set the global random seed for TensorFlow
tf.random.set_seed(974450504)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0 # size=(17, 30, 17, 3), dtype=bool
    
    # var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
    # Conversion: tf.split splits the tensor into a list of tensors along the axis
    var_node_2 = tf.split(var_node_3, num_or_size_splits=3, axis=3)[0] # size=(17, 30, 17, 1), dtype=bool
    
    # var_node_5 = torch.full((17,), 3, dtype=torch.int64)
    # Conversion: tf.fill creates a tensor filled with the specified value
    var_node_5 = tf.fill((17,), 3)
    var_node_5 = tf.cast(var_node_5, tf.int64) # size=(17,), dtype=int64
    
    var_node_6 = arg_1 # size=(15,), dtype=int64
    
    # _input_size_var_node_4 = var_node_5.size(0)
    # Conversion: tf.shape returns the shape tensor of the input
    _input_size_var_node_4 = tf.shape(var_node_5)[0]
    
    # _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    # Conversion: tf.random.uniform generates random integers uniformly from [minval, maxval)
    _index_var_node_4 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_4, dtype=tf.int64)
    
    # var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
    # Conversion: tf.gather gathers slices from params axis according to indices
    var_node_4 = tf.gather(var_node_5, _index_var_node_4) # size=(15,), dtype=int64
    
    # _input_size_var_node_1 = var_node_2.size(0)
    _input_size_var_node_1 = tf.shape(var_node_2)[0]
    
    # _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    _index_var_node_1 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_1, dtype=tf.int64)
    
    # var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
    # Conversion: tf.gather with axis specified is equivalent to index_select
    var_node_1 = tf.gather(var_node_2, _index_var_node_1, axis=0) # size=(15, 30, 17, 1), dtype=bool
    
    # var_node_0 = torch.squeeze(var_node_1)
    # Conversion: tf.squeeze removes dimensions of size 1
    var_node_0 = tf.squeeze(var_node_1) # size=(15, 30, 17), dtype=bool
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    
    # if result.is_complex(): result = result.real
    # Conversion: tf.math.real returns the real part of a complex (or real) tensor
    result = tf.math.real(result)
    
    return result

# Sentinel tensor to ensure gradient computation
# torch.tensor(1.0, requires_grad=True)
# Conversion: Use tf.Variable to track gradients
sentinel = tf.Variable(1.0, dtype=tf.float32)

# arg_0 = torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), (17, 30, 17, 3), (1530, 51, 3, 1))
# Conversion: Generate random data and reshape. as_strided with contiguous strides is equivalent to reshape.
raw_data_0 = tf.random.uniform((26010,), minval=0, maxval=2, dtype=tf.int32)
arg_0 = tf.reshape(tf.cast(raw_data_0, tf.bool), (17, 30, 17, 3))

# arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,))
# Conversion: Generate random data and reshape.
raw_data_1 = tf.random.uniform((15,), minval=5, maxval=30, dtype=tf.int64)
arg_1 = tf.reshape(raw_data_1, (15,))

args = (arg_0, arg_1, sentinel)

# Eager execution
result_original = fuzzed_program(*args)
print('✅ eager success')

# Compiled execution
# torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Conversion: tf.function compiles the Python function into a static graph
compiled_program = tf.function(fuzzed_program, jit_compile=True)
result_compiled = compiled_program(*args)
print('✅ compile success')
```