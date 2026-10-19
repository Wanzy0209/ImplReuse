```python
import tensorflow as tf

# PyTorch: torch._dynamo.config.capture_scalar_outputs = True
# PyTorch: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: PyTorch dynamo configs have no direct equivalent in TensorFlow, ignored.

# PyTorch: torch.manual_seed(1000560)
tf.random.set_seed(1000560)

def fuzzed_program(sentinel):
    # PyTorch: var_node_3 = torch.full((2, 3), 3, dtype=torch.int32)
    var_node_3 = tf.constant(3, shape=(2, 3), dtype=tf.int32) # size=(2, 3), dtype=int32

    # PyTorch: _inp_unique_wide = torch.arange(1, device=var_node_3.device, dtype=torch.int64)
    # Note: Device placement is handled implicitly or via tf.device in TensorFlow
    _inp_unique_wide = tf.range(1, dtype=tf.int64)

    # PyTorch: _uniq_wide = torch.unique(_inp_unique_wide)
    # Note: tf.unique returns (unique_values, indices), we only need the values
    _uniq_wide, _ = tf.unique(_inp_unique_wide)

    # PyTorch: var_node_2 = _uniq_wide.to(var_node_3.dtype)
    var_node_2 = tf.cast(_uniq_wide, var_node_3.dtype) # size=(1,), dtype=int32

    # PyTorch: var_node_1 = torch.reshape(var_node_2, [1])
    var_node_1 = tf.reshape(var_node_2, [1]) # size=(1,), dtype=int32

    # PyTorch: var_node_0 = torch.squeeze(var_node_1)
    var_node_0 = tf.squeeze(var_node_1) # size=(), dtype=int32

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel

    # PyTorch: if result.is_complex(): result = result.real
    # Note: Checking dtype for complex type
    if result.dtype.is_complex:
        result = tf.math.real(result)

    return result

# Sentinel tensor to ensure gradient computation
# PyTorch: sentinel = torch.tensor(1.0, requires_grad=True)
# Note: tf.Variable is used for stateful tensors that require gradients
sentinel = tf.Variable(1.0, dtype=tf.float32)


args = () + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')

# PyTorch: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Note: tf.function compiles the Python function into a TensorFlow graph
compiled_program = tf.function(fuzzed_program)

result_compiled = compiled_program(*args)
print('✅ compile success')
```