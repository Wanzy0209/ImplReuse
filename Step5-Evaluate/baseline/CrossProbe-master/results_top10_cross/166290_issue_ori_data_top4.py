```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# No direct equivalent in TensorFlow; tf.function handles scalar outputs automatically.

# torch.manual_seed(974450504)
tf.random.set_seed(974450504)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0 # size=(17, 30, 17, 3), dtype=bool
    # var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
    # Splits tensor into 3 parts along axis 3 and takes the first part.
    var_node_2 = tf.split(var_node_3, 3, axis=3)[0] # size=(17, 30, 17, 1), dtype=bool
    
    # var_node_5 = torch.full((17,), 3, dtype=torch.int64)
    var_node_5 = tf.constant(3, shape=[17], dtype=tf.int64) # size=(17,), dtype=int64
    
    var_node_6 = arg_1 # size=(15,), dtype=int64
    
    # _input_size_var_node_4 = var_node_5.size(0)
    _input_size_var_node_4 = tf.shape(var_node_5)[0]
    
    # _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    _index_var_node_4 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_4, dtype=tf.int64)
    
    # var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
    var_node_4 = tf.gather(var_node_5, _index_var_node_4, axis=0) # size=(15,), dtype=int64
    
    # _input_size_var_node_1 = var_node_2.size(0)
    _input_size_var_node_1 = tf.shape(var_node_2)[0]
    
    # _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    _index_var_node_1 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_1, dtype=tf.int64)
    
    # var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
    var_node_1 = tf.gather(var_node_2, _index_var_node_1, axis=0) # size=(15, 30, 17, 1), dtype=bool
    
    # var_node_0 = torch.squeeze(var_node_1)
    var_node_0 = tf.squeeze(var_node_1) # size=(15, 30, 17), dtype=bool
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    
    # if result.is_complex():
    if result.dtype.is_complex:
        result = tf.math.real(result)
        
    return result

# Sentinel tensor to ensure gradient computation
# In TensorFlow, gradients are tracked via tf.GradientTape, but we keep the tensor for logic preservation.
sentinel = tf.constant(1.0)

# arg_0 = torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), (17, 30, 17, 3), (1530, 51, 3, 1))
# The stride (1530, 51, 3, 1) indicates a contiguous C-order layout (17*30*17*3 = 26010).
# Therefore, torch.as_strided is equivalent to tf.reshape here.
raw_data_0 = tf.cast(tf.random.uniform((26010,), 0, 2, dtype=tf.int32), tf.bool)
arg_0 = tf.reshape(raw_data_0, (17, 30, 17, 3))

# arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,))
# The stride (1,) for shape (15,) is contiguous. Equivalent to reshape.
raw_data_1 = tf.random.uniform((15,), 5, 30, dtype=tf.int64)
arg_1 = tf.reshape(raw_data_1, (15,))

args = (arg_0, arg_1, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Equivalent to wrapping the function with tf.function.
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```