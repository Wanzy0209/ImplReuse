```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# No direct equivalent in TensorFlow; tf.function handles scalar outputs automatically.

# torch.manual_seed(974450504)
tf.random.set_seed(974450504)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0 # size=(17, 30, 17, 3), stride=(1530, 51, 3, 1), dtype=bool, device=cuda
    # Conversion: torch.chunk -> tf.split
    var_node_2 = tf.split(var_node_3, 3, axis=3)[0] # size=(17, 30, 17, 1), stride=(510, 17, 1, 1), dtype=bool, device=cuda
    # Conversion: torch.full -> tf.constant
    var_node_5 = tf.constant(3, shape=(17,), dtype=tf.int64) # size=(17,), stride=(1,), dtype=int64, device=cuda
    var_node_6 = arg_1 # size=(15,), stride=(1,), dtype=int64, device=cuda
    
    # Conversion: var_node_5.size(0) -> tf.shape
    _input_size_var_node_4 = tf.shape(var_node_5)[0]
    # Conversion: torch.randint -> tf.random.uniform (maxval is exclusive)
    _index_var_node_4 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_4, dtype=tf.int64)
    # Conversion: torch.gather -> tf.gather
    var_node_4 = tf.gather(var_node_5, _index_var_node_4, axis=0) # size=(15,), stride=(1,), dtype=int64, device=cuda
    
    _input_size_var_node_1 = tf.shape(var_node_2)[0]
    _index_var_node_1 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_1, dtype=tf.int64)
    # Conversion: torch.index_select -> tf.gather
    var_node_1 = tf.gather(var_node_2, _index_var_node_1, axis=0) # size=(15, 30, 17, 1), stride=(510, 17, 1, 1), dtype=bool, device=cuda
    # Conversion: torch.squeeze -> tf.squeeze
    var_node_0 = tf.squeeze(var_node_1) # size=(15, 30, 17), stride=(510, 17, 1), dtype=bool, device=cuda
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # Conversion: result.is_complex() -> result.dtype.is_complex
    if result.dtype.is_complex:
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# Conversion: torch.tensor(..., requires_grad=True) -> tf.Variable
sentinel = tf.Variable(1.0, dtype=tf.float32)

# Conversion: torch.as_strided with contiguous strides -> tf.reshape
# The strides (1530, 51, 3, 1) correspond to a standard C-contiguous layout for shape (17, 30, 17, 3)
temp_0 = tf.random.uniform((26010,), minval=0, maxval=2, dtype=tf.int8)
arg_0 = tf.reshape(tf.cast(temp_0, tf.bool), (17, 30, 17, 3))

# Conversion: torch.as_strided with identity strides -> tf.identity or just the tensor
arg_1 = tf.random.uniform((15,), minval=5, maxval=30, dtype=tf.int64)

args = (arg_0, arg_1, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# Conversion: torch.compile -> tf.function with jit_compile=True
compiled_program = tf.function(fuzzed_program, jit_compile=True)
result_compiled = compiled_program(*args)
print('✅ compile success')
```