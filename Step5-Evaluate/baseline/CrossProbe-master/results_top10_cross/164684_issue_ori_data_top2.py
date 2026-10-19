```python
import tensorflow as tf

# PyTorch: torch._dynamo.config.capture_scalar_outputs = True
# PyTorch: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: TensorFlow (tf.function) handles scalar outputs and dynamic shapes automatically.
# No direct equivalent config needed.

# PyTorch: torch.manual_seed(19990)
tf.random.set_seed(19990)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = arg_0 # size=(1,), dtype=bool
    # PyTorch: var_node_1 = torch.squeeze(var_node_2)
    var_node_1 = tf.squeeze(var_node_2) # size=(), dtype=bool

    # PyTorch: var_node_0 = var_node_1.item()
    # Conversion: In TF, we keep the tensor to maintain compatibility with graph mode (tf.function).
    # Using .numpy() would work in eager mode but fail inside tf.function.
    var_node_0 = var_node_1

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel

    # PyTorch: if result.is_complex():
    # Conversion: Check dtype for complex types
    if result.dtype in (tf.complex64, tf.complex128):
        # PyTorch: result = result.real
        # Conversion: Use tf.math.real
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# PyTorch: sentinel = torch.tensor(1.0, requires_grad=True)
# Conversion: Use tf.Variable for mutable state that requires gradients
sentinel = tf.Variable(1.0, dtype=tf.float32)

# PyTorch: arg_0 = torch.randint(0, 2, (1,), dtype=torch.bool) > 0
# Conversion: Generate random integers and cast to boolean
arg_0 = tf.random.uniform((1,), 0, 2, dtype=tf.int32) > 0

args = (arg_0, sentinel)

# Run eager
result_original = fuzzed_program(*args)
print('✅ eager success')

# PyTorch: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Conversion: Wrap in tf.function. fullgraph=True is default behavior for tf.function (tracing).
# dynamic=True is handled by tf.function's re-tracing mechanism for different shapes.
compiled_program = tf.function(fuzzed_program)

result_compiled = compiled_program(*args)
print('✅ compile success')
```