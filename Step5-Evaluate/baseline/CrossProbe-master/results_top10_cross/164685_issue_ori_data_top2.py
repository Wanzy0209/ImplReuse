```python
import tensorflow as tf

# PyTorch: torch._dynamo.config.capture_scalar_outputs = True
# PyTorch: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# TensorFlow does not have direct 1:1 equivalents for these specific Dynamo configs.
# tf.function handles graph compilation and optimization automatically.

# PyTorch: torch.manual_seed(19989)
tf.random.set_seed(19989)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = -6 # dtype=int64 (Python int)
    var_node_3 = arg_0 # dtype=int32 (Python int passed in)
    var_node_1 = var_node_2 * var_node_3 # dtype=int32 (Python int)

    # PyTorch: torch.full((), 1, dtype=torch.int64)
    # TensorFlow: tf.constant creates a scalar tensor
    var_node_5 = tf.constant(1, dtype=tf.int64) # shape=[], dtype=int64

    # PyTorch: var_node_4 = var_node_5.item()
    # In TensorFlow, we keep it as a tensor. TF operations handle Python scalars and Tensors seamlessly.
    var_node_4 = var_node_5

    # PyTorch: var_node_0 = var_node_1 / var_node_4
    # Division of Python int by Tensor results in Tensor
    var_node_0 = var_node_1 / var_node_4

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel

    # PyTorch: if result.is_complex():
    # TensorFlow: Use tf.cond for conditional logic to ensure compatibility with graph mode
    result = tf.cond(
        tf.is_complex(result),
        lambda: tf.math.real(result),
        lambda: result
    )
    return result

# Sentinel tensor to ensure gradient computation
# PyTorch: torch.tensor(1.0, requires_grad=True)
# TensorFlow: tf.Variable is the mutable stateful tensor equivalent
sentinel = tf.Variable(1.0)

# PyTorch: arg_0 = torch.tensor(torch.randn(()), dtype=torch.int32).item()
# TensorFlow: Generate random normal, cast to int32, extract Python int
arg_0 = int(tf.cast(tf.random.normal([]), tf.int32))

args = (arg_0,) + (sentinel,)

# PyTorch: result_original = fuzzed_program(*args)
# Calling the function directly executes it eagerly
result_original = fuzzed_program(*args)
print('✅ eager success')

# PyTorch: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# TensorFlow: tf.function compiles the function into a graph. jit_compile=True enables XLA (similar to fullgraph).
compiled_program = tf.function(fuzzed_program, jit_compile=True)

result_compiled = compiled_program(*args)
print('✅ compile success')
```