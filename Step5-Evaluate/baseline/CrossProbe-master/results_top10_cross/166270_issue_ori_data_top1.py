```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# No direct equivalent needed in TensorFlow 2.x

tf.random.set_seed(1061983224)

def fuzzed_program(arg_0, sentinel):
    # var_node_4 = arg_0
    # torch.chunk(var_node_4, 4, dim=0)[0]
    var_node_3 = tf.split(arg_0, 4, axis=0)[0]
    
    # torch.squeeze(var_node_3)
    var_node_2 = tf.squeeze(var_node_3)
    
    # torch.stack([var_node_2], dim=0)
    var_node_1 = tf.stack([var_node_2], axis=0)
    
    # torch.reshape(var_node_1, [1])
    var_node_0 = tf.reshape(var_node_1, [1])
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    
    # if result.is_complex(): result = result.real
    # TensorFlow requires tf.cond for conditional logic based on tensor values in graph mode
    result = tf.cond(tf.dtypes.is_complex(result), 
                     lambda: tf.math.real(result), 
                     lambda: result)
    return result

# Sentinel tensor to ensure gradient computation
# torch.tensor(1.0, requires_grad=True) -> tf.Variable
sentinel = tf.Variable(1.0)

# torch.as_strided(torch.randint(0, 2, (4,), dtype=torch.int8).bool(), (4,), (1,))
# as_strided with shape (4,) and stride (1,) on a 1D tensor is an identity operation.
# We generate the random tensor and cast it.
raw_rand = tf.random.uniform((4,), minval=0, maxval=2, dtype=tf.int8)
arg_0 = tf.cast(raw_rand, tf.bool)

args = (arg_0, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```