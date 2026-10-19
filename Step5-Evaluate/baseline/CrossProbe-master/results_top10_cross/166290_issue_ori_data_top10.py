```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# No direct equivalent in TensorFlow; tf.function handles scalar outputs automatically.

# torch.manual_seed(974450504)
tf.random.set_seed(974450504)

def fuzzed_program(arg_0, arg_1, sentinel):
    # var_node_3 = arg_0 # size=(17, 30, 17, 3), stride=(1530, 51, 3, 1), dtype=bool, device=cuda
    var_node_3 = arg_0

    # var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0] # size=(17, 30, 17, 1), stride=(510, 17, 1, 1), dtype=bool, device=cuda
    # Conversion: torch.chunk splits tensor into specific number of chunks. tf.split is the equivalent.
    var_node_2 = tf.split(var_node_3, 3, axis=3)[0]

    # var_node_5 = torch.full((17,), 3, dtype=torch.int64) # size=(17,), stride=(1,), dtype=int64, device=cuda
    # Conversion: torch.fill creates a tensor filled with a value. tf.fill is the equivalent.
    var_node_5 = tf.cast(tf.fill((17,), 3), dtype=tf.int64)

    # var_node_6 = arg_1 # size=(15,), stride=(1,), dtype=int64, device=cuda
    var_node_6 = arg_1

    # _input_size_var_node_4 = var_node_5.size(0)
    _input_size_var_node_4 = tf.shape(var_node_5)[0]

    # _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    # Conversion: torch.randint generates random integers. tf.random.uniform is the equivalent.
    # Note: PyTorch randint is exclusive on the upper bound, similar to tf.random.uniform.
    _index_var_node_4 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_4, dtype=tf.int64)

    # var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4) # size=(15,), stride=(1,), dtype=int64, device=cuda
    # Conversion: torch.gather selects values along an axis. tf.gather is the equivalent.
    var_node_4 = tf.gather(var_node_5, _index_var_node_4, axis=0)

    # _input_size_var_node_1 = var_node_2.size(0)
    _input_size_var_node_1 = tf.shape(var_node_2)[0]

    # _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    _index_var_node_1 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_1, dtype=tf.int64)

    # var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1) # size=(15, 30, 17, 1), stride=(510, 17, 1, 1), dtype=bool, device=cuda
    # Conversion: torch.index_select selects values along an axis using indices. tf.gather is the equivalent.
    var_node_1 = tf.gather(var_node_2, _index_var_node_1, axis=0)

    # var_node_0 = torch.squeeze(var_node_1) # size=(15, 30, 17), stride=(510, 17, 1), dtype=bool, device=cuda
    # Conversion: torch.squeeze removes dimensions of size 1. tf.squeeze is the equivalent.
    var_node_0 = tf.squeeze(var_node_1)

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel

    # if result.is_complex(): result = result.real
    # Conversion: Check if dtype is complex and take real part if so.
    # Since var_node_0 is bool and sentinel is float, result is float, but we preserve the logic.
    def take_real(): return tf.math.real(result)
    def identity(): return result
    result = tf.cond(tf.dtypes.is_complex(result), take_real, identity)

    return result

# Sentinel tensor to ensure gradient computation
# torch.tensor(1.0, requires_grad=True)
# Conversion: tf.Variable is used for state that requires gradients.
sentinel = tf.Variable(1.0, dtype=tf.float32)

# arg_0 = torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), (17, 30, 17, 3), (1530, 51, 3, 1))
# Conversion: The as_strided call here creates a contiguous view, equivalent to reshape.
# torch.randint(0, 2, ...) -> tf.random.uniform(..., 0, 2)
# .bool() -> tf.cast(..., tf.bool)
raw_arg_0 = tf.cast(tf.random.uniform((26010,), 0, 2, dtype=tf.int32), tf.bool)
arg_0 = tf.reshape(raw_arg_0, (17, 30, 17, 3))

# arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,))
# Conversion: Identity reshape.
raw_arg_1 = tf.cast(tf.random.uniform((15,), 5, 30, dtype=tf.int32), tf.int64)
arg_1 = tf.reshape(raw_arg_1, (15,))

args = (arg_0, arg_1, sentinel)

# Run eager
result_original = fuzzed_program(*args)
print('✅ eager success')

# Compile
# torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Conversion: tf.function compiles a Python function into a callable TensorFlow graph.
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```