```python
import tensorflow as tf

# torch._dynamo.config... -> No direct equivalent, TF handles compilation via tf.function
# torch.manual_seed(114503) -> tf.random.set_seed
tf.random.set_seed(114503)

def fuzzed_program(arg_0, arg_1, sentinel):
    # var_node_4 = torch.full((2, 3), 3, dtype=torch.int16)
    var_node_4 = tf.fill((2, 3), tf.cast(3, tf.int16)) # size=(2, 3), dtype=int16

    # var_node_3 = torch.unique(var_node_4)
    # tf.unique returns (unique, idx), we take unique
    var_node_3, _ = tf.unique(var_node_4) # size=(1,), dtype=int16

    # var_node_2 = torch.squeeze(var_node_3)
    var_node_2 = tf.squeeze(var_node_3) # size=(), dtype=int16

    var_node_7 = arg_0 # size=(), dtype=int16
    var_node_8 = arg_1 # size=(), dtype=int16

    # var_node_6 = torch.sub(var_node_7, var_node_8)
    var_node_6 = tf.subtract(var_node_7, var_node_8) # size=(), dtype=int16

    # var_node_10 = torch.full((1,), 3, dtype=torch.int16)
    var_node_10 = tf.fill((1,), tf.cast(3, tf.int16)) # size=(1,), dtype=int16

    # var_node_9 = torch.squeeze(var_node_10)
    var_node_9 = tf.squeeze(var_node_10) # size=(), dtype=int16

    # var_node_5 = torch.add(var_node_6, var_node_9)
    var_node_5 = tf.add(var_node_6, var_node_9) # size=(), dtype=int16

    # var_node_1 = torch.div(var_node_2, var_node_5)
    # PyTorch div promotes to float. TF / does too.
    var_node_1 = tf.divide(var_node_2, var_node_5) # size=(), dtype=float

    # var_node_0 = var_node_1.item()
    # Conversion: .item() extracts Python scalar. In TF graph mode, we keep as 0-d tensor.
    var_node_0 = var_node_1

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel

    # if result.is_complex(): result = result.real
    # Conversion: Use tf.cond for graph-compatible conditional logic
    result = tf.cond(tf.math.is_complex(result),
                     lambda: tf.math.real(result),
                     lambda: result)
    return result

# Sentinel tensor to ensure gradient computation
# torch.tensor(1.0, requires_grad=True) -> tf.Variable
sentinel = tf.Variable(1.0)

# arg_0 = torch.as_strided(torch.randint(5, 30, (1,)).to(torch.int16), (), ())
# Conversion: torch.randint -> tf.random.uniform, torch.as_strided -> tf.reshape
arg_0 = tf.reshape(tf.cast(tf.random.uniform((1,), 5, 30, dtype=tf.int32), tf.int16), [])
arg_1 = tf.reshape(tf.cast(tf.random.uniform((1,), 5, 30, dtype=tf.int32), tf.int16), [])

args = (arg_0, arg_1, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Conversion: torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)

result_compiled = compiled_program(*args)
print('✅ compile success')
```