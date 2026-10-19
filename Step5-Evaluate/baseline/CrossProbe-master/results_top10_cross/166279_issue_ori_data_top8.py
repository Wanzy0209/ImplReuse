```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# No direct equivalent in TensorFlow, graph execution handles scalar outputs automatically.

# torch.manual_seed(1166094474)
tf.random.set_seed(1166094474)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    # var_node_3 = torch.full((12,), False, dtype=torch.bool)
    # Conversion: tf.fill creates a tensor filled with a value, cast to bool
    var_node_3 = tf.cast(tf.fill([12], False), tf.bool) # size=(12,)

    # var_node_2 = torch.chunk(var_node_3, 4, dim=0)[0]
    # Conversion: tf.split divides the tensor into num_splits parts along axis
    var_node_2 = tf.split(var_node_3, 4, axis=0)[0] # size=(3,)

    var_node_6 = arg_0 # size=(12,)
    var_node_7 = arg_1 # size=(10,)

    # _input_size_var_node_5 = var_node_6.size(0)
    # Conversion: tf.shape returns the shape tensor
    _input_size_var_node_5 = tf.shape(var_node_6)[0]

    # _index_var_node_5 = torch.randint(0, _input_size_var_node_5, (10,), device=var_node_6.device)
    # Conversion: tf.random.uniform generates random values, maxval is exclusive like PyTorch
    _index_var_node_5 = tf.random.uniform([10], minval=0, maxval=_input_size_var_node_5, dtype=tf.int64)

    # var_node_5 = torch.gather(var_node_6, 0, _index_var_node_5)
    # Conversion: tf.gather gathers values along an axis
    var_node_5 = tf.gather(var_node_6, _index_var_node_5, axis=0) # size=(10,)

    # var_node_4 = torch.chunk(var_node_5, 2, dim=0)[0]
    var_node_4 = tf.split(var_node_5, 2, axis=0)[0] # size=(5,)

    var_node_10 = arg_2 # size=(6, 4)

    # var_node_9 = torch.chunk(var_node_10, 4, dim=1)[0]
    var_node_9 = tf.split(var_node_10, 4, axis=1)[0] # size=(6, 1)

    # var_node_8 = torch.squeeze(var_node_9)
    # Conversion: tf.squeeze removes dimensions of size 1
    var_node_8 = tf.squeeze(var_node_9) # size=(6,)

    # var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0)
    # Conversion: tf.concat concatenates tensors along an axis
    var_node_1 = tf.concat([var_node_2, var_node_4, var_node_8], axis=0) # size=(14,)

    var_node_11 = arg_3 # size=(2,)

    # var_node_0 = torch.cat([var_node_1, var_node_11], dim=0)
    var_node_0 = tf.concat([var_node_1, var_node_11], axis=0) # size=(16,)

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel

    # if result.is_complex():
    # Conversion: Check dtype property
    if result.dtype.is_complex:
        result = tf.math.real(result)
    
    return result

# Sentinel tensor to ensure gradient computation
# torch.tensor(1.0, requires_grad=True)
# Conversion: tf.Variable is used for state and gradients
sentinel = tf.Variable(1.0, dtype=tf.float32)

# arg_0 = torch.as_strided(torch.randint(0, 2, (12,), dtype=torch.int8).bool(), (12,), (1,))
# Conversion: Create random bool tensor directly. Stride (1,) is default for 1D.
arg_0 = tf.cast(tf.random.uniform((12,), 0, 2, dtype=tf.int32), tf.bool)

# arg_1 = torch.as_strided(torch.randint(5, 30, (10,)).to(torch.int64), (10,), (1,))
# Conversion: Create random int64 tensor.
arg_1 = tf.random.uniform((10,), 5, 30, dtype=tf.int64)

# arg_2 = torch.as_strided(torch.randint(0, 2, (24,), dtype=torch.int8).bool(), (6, 4), (4, 1))
# Conversion: Create random bool tensor and reshape. Stride (4, 1) implies row-major, which is TF default.
arg_2 = tf.reshape(tf.cast(tf.random.uniform((24,), 0, 2, dtype=tf.int32), tf.bool), (6, 4))

# arg_3 = torch.as_strided(torch.randint(0, 2, (2,), dtype=torch.int8).bool(), (2,), (1,))
arg_3 = tf.cast(tf.random.uniform((2,), 0, 2, dtype=tf.int32), tf.bool)

args = (arg_0, arg_1, arg_2, arg_3, sentinel)

# Eager execution
result_original = fuzzed_program(*args)
print('✅ eager success')

# Compiled execution
# torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Conversion: tf.function compiles the Python function into a TensorFlow graph
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```