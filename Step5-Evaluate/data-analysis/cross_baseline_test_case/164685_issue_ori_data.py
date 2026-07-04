```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: PyTorch dynamo configs are specific to torch.compile and have no direct equivalent in TensorFlow.

tf.random.set_seed(19989)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = -6 # dtype=int64
    var_node_3 = arg_0 # dtype=int32
    var_node_1 = var_node_2 * var_node_3 # dtype=int32
    
    # torch.full((), 1, dtype=torch.int64)
    var_node_5 = tf.constant(1, dtype=tf.int64) # size=(), dtype=int64
    
    # var_node_4 = var_node_5.item()
    # In TensorFlow graph mode (tf.function), we keep operations as tensors.
    # Since var_node_5 is a constant 1, var_node_4 is effectively 1.
    var_node_4 = var_node_5 
    
    var_node_0 = var_node_1 / var_node_4 # dtype=float (promoted from int)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    
    # if result.is_complex():
    # TensorFlow tensors have a dtype attribute that can be checked.
    if result.dtype.is_complex:
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# torch.tensor(1.0, requires_grad=True)
# In TensorFlow, a tf.Variable is used for stateful tensors that might require gradients.
sentinel = tf.Variable(1.0, dtype=tf.float32)

# arg_0 = torch.tensor(torch.randn(()), dtype=torch.int32).item()
# torch.randn(()) -> tf.random.normal(())
# torch.tensor(..., dtype=int32) -> tf.cast(..., tf.int32)
# .item() -> .numpy().item() (extracts Python scalar)
raw_rand = tf.random.normal(())
arg_0 = int(tf.cast(raw_rand, tf.int32).numpy())

args = (arg_0,) + (sentinel,)

# Eager execution
result_original = fuzzed_program(*args)
print('✅ eager success')

# Compile
# torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# tf.function is the TensorFlow equivalent for graph compilation/optimization.
# jit_compile=True provides a closer equivalent to fullgraph compilation.
compiled_program = tf.function(fuzzed_program, jit_compile=True)
result_compiled = compiled_program(*args)
print('✅ compile success')
```