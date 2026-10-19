```python
import tensorflow as tf

# PyTorch specific configurations omitted:
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Set random seed
tf.random.set_seed(19990)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = arg_0 # size=(1,), dtype=bool
    var_node_1 = tf.squeeze(var_node_2) # size=(), dtype=bool
    # var_node_0 = var_node_1.item()
    # In TensorFlow, 0-d tensors can be used directly in arithmetic operations
    var_node_0 = var_node_1
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # if result.is_complex():
    # TensorFlow uses tf.math.is_complex to check if the tensor's dtype is complex
    if tf.math.is_complex(result):
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# tf.Variable is used for tensors that require gradients
sentinel = tf.Variable(1.0)

# arg_0 = torch.randint(0, 2, (1,), dtype=torch.bool) > 0
# Equivalent: generate random integers 0 or 1, cast to bool
arg_0 = tf.cast(tf.random.uniform((1,), 0, 2, dtype=tf.int32), tf.bool)

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
# torch.compile equivalent
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```