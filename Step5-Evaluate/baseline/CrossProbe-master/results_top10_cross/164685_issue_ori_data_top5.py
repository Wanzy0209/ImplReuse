```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: PyTorch dynamo configs have no direct equivalent in TensorFlow.

# torch.manual_seed(19989)
tf.random.set_seed(19989)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = -6 # dtype=int64
    var_node_3 = arg_0 # dtype=int32
    var_node_1 = var_node_2 * var_node_3 # dtype=int32
    # torch.full((), 1, dtype=torch.int64)
    var_node_5 = tf.constant(1, dtype=tf.int64) # size=(), dtype=int64
    # var_node_4 = var_node_5.item() -> Kept as tensor for graph compatibility
    var_node_4 = var_node_5 # dtype=int64
    var_node_0 = var_node_1 / var_node_4 # dtype=float (truediv)
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # if result.is_complex():
    # TensorFlow requires tf.cond for conditional logic in graph mode
    result = tf.cond(tf.math.is_complex(result),
                     lambda: tf.math.real(result),
                     lambda: result)
    return result

# Sentinel tensor to ensure gradient computation
# torch.tensor(1.0, requires_grad=True)
sentinel = tf.Variable(1.0, dtype=tf.float32)

# arg_0 = torch.tensor(torch.randn(()), dtype=torch.int32).item()
arg_0 = tf.cast(tf.random.normal([]), tf.int32).numpy().item()

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
# torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```