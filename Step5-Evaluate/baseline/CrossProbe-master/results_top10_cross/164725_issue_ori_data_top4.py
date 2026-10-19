```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: PyTorch dynamo configs are not applicable in TensorFlow.

tf.random.set_seed(1000030)

def fuzzed_program(sentinel):
    # var_node_4 = torch.full((6,), True, dtype=torch.bool)
    var_node_4 = tf.fill([6], True) # size=(6,), dtype=bool

    # var_node_3 = torch.reshape(var_node_4, [2, 3])
    var_node_3 = tf.reshape(var_node_4, [2, 3]) # size=(2, 3), dtype=bool

    # var_node_5 = torch.full((2, 3), False, dtype=torch.bool)
    var_node_5 = tf.fill([2, 3], False) # size=(2, 3), dtype=bool

    # _x_ms = torch.arange(max(1, 1), device=var_node_3.device).to(var_node_3.dtype)
    # max(1, 1) is 1. torch.arange(1) is [0]. Cast to bool -> [False].
    _x_ms = tf.cast(tf.range(1), dtype=tf.bool) # size=(1,), dtype=bool

    # _mask_ms = torch.zeros_like(_x_ms, dtype=torch.bool)
    _mask_ms = tf.zeros_like(_x_ms, dtype=torch.bool) # size=(1,), dtype=bool

    # _mask_ms[:1] = True
    # TF tensors are immutable, use tensor_scatter_nd_update for in-place modification logic
    indices = tf.constant([[0]])
    updates = tf.constant([True])
    _mask_ms = tf.tensor_scatter_nd_update(_mask_ms, indices, updates) # size=(1,), dtype=bool

    # var_node_2 = torch.masked_select(_x_ms, _mask_ms)
    var_node_2 = tf.boolean_mask(_x_ms, _mask_ms) # size=(1,), dtype=bool

    # var_node_1 = torch.squeeze(var_node_2)
    var_node_1 = tf.squeeze(var_node_2) # size=(), dtype=bool

    # var_node_0 = var_node_1.item()
    # In TF, to maintain graph compatibility (especially for tf.function),
    # we keep it as a 0-d tensor rather than converting to a Python scalar.
    var_node_0 = var_node_1 # size=(), dtype=bool

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel

    # if result.is_complex():
    if tf.dtypes.is_complex(result.dtype):
        # result = result.real
        result = tf.math.real(result)
    
    return result

# Sentinel tensor to ensure gradient computation
# torch.tensor(1.0, requires_grad=True)
sentinel = tf.Variable(1.0, dtype=tf.float32)


args = () + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')

# compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Using tf.function for graph compilation. jit_compile=True is closer to fullgraph.
compiled_program = tf.function(fuzzed_program, jit_compile=True)

result_compiled = compiled_program(*args)
print('✅ compile success')
```