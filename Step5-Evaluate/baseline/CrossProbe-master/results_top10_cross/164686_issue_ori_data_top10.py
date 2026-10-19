```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: PyTorch dynamo configs have no direct equivalent in TensorFlow

# torch.manual_seed(13653)
tf.random.set_seed(13653)

def fuzzed_program(arg_0, arg_1, sentinel):
    # var_node_3 = torch.full((), 1.0, dtype=torch.float32)
    # Conversion: torch.full -> tf.constant for scalar creation
    var_node_3 = tf.constant(1.0, dtype=tf.float32) # size=(), dtype=float32

    # var_node_2 = var_node_3.item()
    # Conversion: .item() -> .numpy().item() to extract Python scalar
    var_node_2 = var_node_3.numpy().item() # dtype=float32

    var_node_5 = -3 # dtype=int32
    var_node_6 = arg_0 # dtype=int64
    var_node_4 = var_node_5 + var_node_6 # dtype=int32
    var_node_1 = var_node_2 + var_node_4 # dtype=float32
    var_node_9 = 1 # dtype=int64
    var_node_10 = -10 # dtype=int32
    var_node_8 = var_node_9 / var_node_10 # dtype=int64
    var_node_12 = arg_1 # dtype=int64
    var_node_13 = -5 # dtype=int32
    var_node_11 = var_node_12 / var_node_13 # dtype=int32
    var_node_7 = var_node_8 + var_node_11 # dtype=int32
    var_node_0 = var_node_1 * var_node_7 # dtype=float32

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel

    # if result.is_complex()
    # Conversion: Tensor.is_complex() -> dtype.is_complex
    if result.dtype.is_complex:
        # result = result.real
        # Conversion: Tensor.real -> tf.math.real
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# torch.tensor(1.0, requires_grad=True)
# Conversion: Use tf.Variable to mimic a stateful tensor that can track gradients
sentinel = tf.Variable(1.0, dtype=tf.float32)

# arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
# Conversion: torch.randn -> tf.random.normal, cast to int64, then extract scalar
arg_0 = int(tf.cast(tf.random.normal([]), tf.int64).numpy())
arg_1 = int(tf.cast(tf.random.normal([]), tf.int64).numpy())

args = (arg_0, arg_1, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Conversion: torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```