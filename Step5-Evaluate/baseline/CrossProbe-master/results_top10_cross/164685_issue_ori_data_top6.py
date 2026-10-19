```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: PyTorch Dynamo configs are specific to torch.compile backend.
# TensorFlow's tf.function handles graph capture automatically.

tf.random.set_seed(19989)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = -6 # dtype=int64
    var_node_3 = arg_0 # dtype=int32
    var_node_1 = var_node_2 * var_node_3 # dtype=int32
    
    # torch.full((), 1, dtype=torch.int64) -> tf.constant
    var_node_5 = tf.constant(1, dtype=tf.int64) # shape=(), dtype=int64
    
    # var_node_4 = var_node_5.item() 
    # Conversion: In TF, we keep it as a tensor for graph compatibility.
    # Division works with tensors directly.
    var_node_4 = var_node_5 # dtype=int64
    
    var_node_0 = var_node_1 / var_node_4 # dtype=float (implicit promotion)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    
    # if result.is_complex() -> tf.math.is_complex
    if tf.math.is_complex(result):
        # result = result.real -> tf.math.real
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# requires_grad=True implies a Variable in TF context for gradient tracking
sentinel = tf.Variable(1.0, dtype=tf.float32)

# arg_0 = torch.tensor(torch.randn(()), dtype=torch.int32).item()
# Conversion: Generate random normal, cast to int32, extract to python int
arg_0 = int(tf.cast(tf.random.normal([]), tf.int32).numpy())

args = (arg_0,) + (sentinel,)

# Eager execution
result_original = fuzzed_program(*args)
print('✅ eager success')

# torch.compile(...) -> tf.function with jit_compile=True
compiled_program = tf.function(fuzzed_program, jit_compile=True)
result_compiled = compiled_program(*args)
print('✅ compile success')
```