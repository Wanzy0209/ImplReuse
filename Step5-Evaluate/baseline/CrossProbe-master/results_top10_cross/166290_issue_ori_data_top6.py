```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# Note: TensorFlow does not have a direct equivalent for this specific Dynamo config.

# torch.manual_seed(974450504)
tf.random.set_seed(974450504)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0 # size=(17, 30, 17, 3), stride=(1530, 51, 3, 1), dtype=bool, device=cuda
    # var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
    # Conversion: tf.split returns a list of tensors.
    var_node_2 = tf.split(var_node_3, 3, axis=3)[0] # size=(17, 30, 17, 1), stride=(510, 17, 1, 1), dtype=bool, device=cuda
    # var_node_5 = torch.full((17,), 3, dtype=torch.int64)
    # Conversion: tf.constant with shape and dtype creates a filled tensor.
    var_node_5 = tf.constant(3, shape=(17,), dtype=tf.int64) # size=(17,), stride=(1,), dtype=int64, device=cuda
    var_node_6 = arg_1 # size=(15,), stride=(1,), dtype=int64, device=cuda
    # _input_size_var_node_4 = var_node_5.size(0)
    # Conversion: tf.shape returns a 1-D tensor representing the shape.
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
    # Conversion: tf.gather is the equivalent of index_select along an axis.
    var_node_1 = tf.gather(var_node_2, _index_var_node_1, axis=0) # size=(15, 30, 17, 1), stride=(510, 17, 1, 1), dtype=bool, device=cuda
    # var_node_0 = torch.squeeze(var_node_1)
    # Conversion: tf.squeeze removes dimensions of size 1.
    var_node_0 = tf.squeeze(var_node_1) # size=(15, 30, 17), stride=(510, 17, 1), dtype=bool, device=cuda
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # if result.is_complex():
    # Conversion: tf.dtypes.is_complex checks if the dtype is complex.
    if tf.dtypes.is_complex(result.dtype):
        # result = result.real
        # Conversion: tf.math.real returns the real part of a complex tensor.
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# sentinel = torch.tensor(1.0, requires_grad=True)
# Conversion: tf.Variable is used for state/mutable tensors, similar to requires_grad=True tensors in PyTorch graphs.
sentinel = tf.Variable(1.0, dtype=tf.float32)

# arg_0 = torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), (17, 30, 17, 3), (1530, 51, 3, 1))
# Conversion: torch.as_strided with contiguous strides is equivalent to reshape.
# We generate random ints, cast to bool, and reshape.
arg_0 = tf.reshape(tf.cast(tf.random.uniform((26010,), minval=0, maxval=2, dtype=tf.int32), tf.bool), (17, 30, 17, 3))

# arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,))
# Conversion: Generating random int64 tensor directly.
arg_1 = tf.random.uniform((15,), minval=5, maxval=30, dtype=tf.int64)

args = (arg_0, arg_1, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')
# compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Conversion: tf.function compiles a Python function into a callable TensorFlow graph.
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```