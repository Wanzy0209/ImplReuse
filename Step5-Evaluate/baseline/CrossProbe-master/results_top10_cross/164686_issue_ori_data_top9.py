```python
import tensorflow as tf

# PyTorch: torch._dynamo.config.capture_scalar_outputs = True
# PyTorch: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: torch._dynamo configs are specific to PyTorch 2.0+ compilation.
# TensorFlow's tf.function handles graph compilation and scalar outputs automatically.

# PyTorch: torch.manual_seed(13653)
tf.random.set_seed(13653)

def fuzzed_program(arg_0, arg_1, sentinel):
    # PyTorch: torch.full((), 1.0, dtype=torch.float32)
    var_node_3 = tf.fill((), 1.0) # size=(), dtype=float32
    
    # PyTorch: var_node_3.item()
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
    
    # PyTorch: if result.is_complex()
    # Conversion: In TensorFlow, we check the dtype property for complex types in Python control flow
    if result.dtype.is_complex:
        # PyTorch: result.real
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# PyTorch: torch.tensor(1.0, requires_grad=True)
# Conversion: tf.Variable is used for stateful tensors that require gradients
sentinel = tf.Variable(1.0, dtype=tf.float32)

# PyTorch: torch.tensor(torch.randn(()), dtype=torch.int64).item()
# Conversion: tf.random.normal generates float, cast to int64, then extract scalar
arg_0 = tf.cast(tf.random.normal(()), tf.int64).numpy().item()
arg_1 = tf.cast(tf.random.normal(()), tf.int64).numpy().item()

args = (arg_0, arg_1, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# PyTorch: torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Conversion: tf.function compiles the Python function into a static graph
compiled_program = tf.function(fuzzed_program)

result_compiled = compiled_program(*args)
print('✅ compile success')
```