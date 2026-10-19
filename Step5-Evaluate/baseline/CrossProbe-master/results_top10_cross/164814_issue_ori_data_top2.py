```python
import tensorflow as tf

# PyTorch Dynamo specific configs are not directly translatable to TF.
# TF's tf.function handles graph compilation automatically.
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Set random seed
tf.random.set_seed(1000560)

def fuzzed_program(sentinel):
    # torch.full((2, 3), 3, dtype=torch.int32)
    var_node_3 = tf.constant(3, shape=(2, 3), dtype=tf.int32)

    # torch.arange(1, device=var_node_3.device, dtype=torch.int64)
    # TF handles device placement implicitly or via context managers.
    _inp_unique_wide = tf.range(1, dtype=tf.int64)

    # torch.unique(_inp_unique_wide)
    # tf.unique returns a tuple (unique_values, indices), so we select the first element.
    _uniq_wide = tf.unique(_inp_unique_wide)[0]

    # _uniq_wide.to(var_node_3.dtype)
    var_node_2 = tf.cast(_uniq_wide, tf.int32)

    # torch.reshape(var_node_2, [1])
    var_node_1 = tf.reshape(var_node_2, [1])

    # torch.squeeze(var_node_1)
    var_node_0 = tf.squeeze(var_node_1)

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel

    # if result.is_complex():
    # In TF, we check the dtype object directly.
    if tf.dtypes.is_complex(result.dtype):
        # result = result.real
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# torch.tensor(1.0, requires_grad=True)
# In TF, Variables are used for state and gradients.
sentinel = tf.Variable(1.0)

args = () + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')

# compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# tf.function is the TF equivalent for graph compilation
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```