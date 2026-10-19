```python
import tensorflow as tf

# PyTorch: torch._dynamo.config.capture_scalar_outputs = True
# PyTorch: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: These are PyTorch specific configurations. TensorFlow's tf.function handles scalar outputs and dynamic shapes by default.

# PyTorch: torch.manual_seed(13653)
# TensorFlow: Set the global random seed
tf.random.set_seed(13653)

def fuzzed_program(arg_0, arg_1, sentinel):
    # PyTorch: var_node_3 = torch.full((), 1.0, dtype=torch.float32)
    # TensorFlow: tf.fill creates a tensor filled with the given value
    var_node_3 = tf.fill((), 1.0) # size=(), dtype=float32

    # PyTorch: var_node_2 = var_node_3.item()
    # TensorFlow: Convert tensor to numpy scalar
    var_node_2 = var_node_3.numpy().item() # dtype=float32

    var_node_5 = -3 # dtype=int32
    var_node_6 = arg_0 # dtype=int64
    var_node_4 = var_node_5 + var_node_6 # dtype=int32
    var_node_1 = var_node_2 + var_node_4 # dtype=float32
    var_node_9 = 1 # dtype=int64
    var_node_10 = -10 # dtype=int32
    var_node_8 = var_node_9 / var_node_10 # dtype=float64 (Python division)
    var_node_12 = arg_1 # dtype=int64
    var_node_13 = -5 # dtype=int32
    var_node_11 = var_node_12 / var_node_13 # dtype=float64 (Python division)
    var_node_7 = var_node_8 + var_node_11 # dtype=float64
    var_node_0 = var_node_1 * var_node_7 # dtype=float64

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    
    # PyTorch: if result.is_complex(): result = result.real
    # TensorFlow: Check if dtype is complex
    if tf.dtypes.complex(result.dtype):
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# PyTorch: sentinel = torch.tensor(1.0, requires_grad=True)
# TensorFlow: Use tf.Variable to mimic a mutable tensor with gradient tracking potential
sentinel = tf.Variable(1.0, dtype=tf.float32)

# PyTorch: arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
# TensorFlow: Generate random normal, cast to int64, extract to python scalar
arg_0 = tf.cast(tf.random.normal(()), tf.int64).numpy().item()

# PyTorch: arg_1 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
arg_1 = tf.cast(tf.random.normal(()), tf.int64).numpy().item()

args = (arg_0, arg_1, sentinel)

# PyTorch: result_original = fuzzed_program(*args)
# TensorFlow: Call the function directly (eager execution)
result_original = fuzzed_program(*args)
print('✅ eager success')

# PyTorch: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# TensorFlow: Use tf.function to compile the graph. autograph=True handles dynamic control flow.
compiled_program = tf.function(fuzzed_program, autograph=True)

# PyTorch: result_compiled = compiled_program(*args)
result_compiled = compiled_program(*args)
print('✅ compile success')
```