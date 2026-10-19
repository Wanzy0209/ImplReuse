```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: TensorFlow (tf.function) handles scalar outputs and dynamic shapes automatically.

tf.random.set_seed(1000560)

def fuzzed_program(sentinel):
    # var_node_3 = torch.full((2, 3), 3, dtype=torch.int32)
    var_node_3 = tf.constant(3, shape=(2, 3), dtype=tf.int32) # size=(2, 3), dtype=int32

    # _inp_unique_wide = torch.arange(1, device=var_node_3.device, dtype=torch.int64)
    # Note: Device placement is handled implicitly in TensorFlow
    _inp_unique_wide = tf.range(1, dtype=tf.int64)

    # _uniq_wide = torch.unique(_inp_unique_wide)
    # tf.unique returns a named tuple, .y contains the unique values
    _uniq_wide = tf.unique(_inp_unique_wide).y

    # var_node_2 = _uniq_wide.to(var_node_3.dtype)
    var_node_2 = tf.cast(_uniq_wide, tf.int32) # size=(1,), dtype=int32

    # var_node_1 = torch.reshape(var_node_2, [1])
    var_node_1 = tf.reshape(var_node_2, [1]) # size=(1,), dtype=int32

    # var_node_0 = torch.squeeze(var_node_1)
    var_node_0 = tf.squeeze(var_node_1) # size=(), dtype=int32

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel

    # if result.is_complex(): result = result.real
    # tf.math.real works on both complex and real tensors
    result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# Using tf.Variable to mimic requires_grad=True behavior
sentinel = tf.Variable(1.0, dtype=tf.float32)


args = () + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
# tf.function is the equivalent of torch.compile
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```