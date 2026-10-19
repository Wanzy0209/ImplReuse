```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Conversion: PyTorch Dynamo configs are not applicable in TensorFlow.

tf.random.set_seed(19989)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = -6 # dtype=int64
    var_node_3 = arg_0 # dtype=int32
    var_node_1 = var_node_2 * var_node_3 # dtype=int32
    # Conversion: torch.full((), 1, dtype=torch.int64) -> tf.constant
    var_node_5 = tf.constant(1, dtype=tf.int64) # size=(), stride=(), dtype=int64, device=cuda
    # Conversion: .item() extracts Python scalar
    var_node_4 = var_node_5.numpy().item() # dtype=int64
    var_node_0 = var_node_1 / var_node_4 # dtype=int64
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    # Conversion: result.is_complex() -> result.dtype.is_complex
    if result.dtype.is_complex:
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
# Conversion: requires_grad=True -> tf.Variable
sentinel = tf.Variable(1.0)

# Conversion: torch.randn -> tf.random.normal
# Conversion: .item() -> .numpy().item()
arg_0 = tf.cast(tf.random.normal(()), dtype=tf.int32).numpy().item()

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
# Conversion: torch.compile -> tf.function with jit_compile=True
compiled_program = tf.function(fuzzed_program, jit_compile=True)
result_compiled = compiled_program(*args)
print('✅ compile success')
```