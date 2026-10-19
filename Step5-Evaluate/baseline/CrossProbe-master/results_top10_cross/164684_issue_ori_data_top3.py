```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: PyTorch dynamo configurations are not applicable in TensorFlow.

tf.random.set_seed(19990)

def fuzzed_program(arg_0, sentinel):
    # var_node_2 = arg_0 # size=(1,), stride=(1,), dtype=bool, device=cuda
    # var_node_1 = torch.squeeze(var_node_2) # size=(), stride=(), dtype=bool, device=cuda
    var_node_1 = tf.squeeze(arg_0) # shape=(), dtype=bool

    # var_node_0 = var_node_1.item() # dtype=bool
    # In TensorFlow, we keep it as a 0-d tensor to maintain compatibility with graph execution (tf.function)
    var_node_0 = var_node_1

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel

    # if result.is_complex():
    #     result = result.real
    # TensorFlow equivalent for checking complex dtype
    if result.dtype.is_complex:
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# torch.tensor(1.0, requires_grad=True) -> tf.Variable
sentinel = tf.Variable(1.0, dtype=tf.float32)

# arg_0 = torch.randint(0, 2, (1,), dtype=torch.bool) > 0
# TensorFlow equivalent: generate random integers and cast to boolean
arg_0 = tf.cast(tf.random.uniform((1,), 0, 2, dtype=tf.int32), tf.bool)

args = (arg_0, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# TensorFlow equivalent is tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```