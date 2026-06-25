```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: PyTorch dynamo configs are not applicable in TensorFlow.

# torch.manual_seed(1012969)
tf.random.set_seed(1012969)

def fuzzed_program(arg_0, arg_1, sentinel):
    # var_node_3 = arg_0 # size=(2, 10), stride=(10, 1), dtype=float64, device=cuda
    # var_node_4 = arg_1 # size=(10, 3), stride=(3, 1), dtype=float64, device=cuda
    # var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_4.to(torch.float64))
    # Conversion: Explicit cast to float64, then matmul.
    var_node_2 = tf.matmul(tf.cast(arg_0, tf.float64), tf.cast(arg_1, tf.float64)) # size=(2, 3), dtype=float64

    # _inp_unique_wide = torch.arange(1, device=var_node_2.device, dtype=torch.int64)
    # Conversion: torch.arange(1) is [0]. tf.range(1) is [0, 1]. tf.range(0, 1) is [0].
    _inp_unique_wide = tf.range(0, 1, dtype=tf.int64)

    # _uniq_wide = torch.unique(_inp_unique_wide)
    # Conversion: tf.unique returns a named tuple, we need the 'y' field for values.
    _uniq_wide = tf.unique(_inp_unique_wide).y

    # var_node_1 = _uniq_wide.to(var_node_2.dtype)
    # Conversion: Cast to float64.
    var_node_1 = tf.cast(_uniq_wide, tf.float64) # size=(1,), dtype=float64

    # var_node_5 = torch.full((1, 18), 0.40330381448978797, dtype=torch.float64)
    # Conversion: tf.fill with a constant value.
    var_node_5 = tf.fill((1, 18), tf.constant(0.40330381448978797, dtype=tf.float64)) # size=(1, 18), dtype=float64

    # var_node_0 = torch.matmul(var_node_1.to(torch.float64), var_node_5.to(torch.float64))
    # Conversion: var_node_1 is (1,), var_node_5 is (1, 18).
    # PyTorch matmul broadcasts (1,) to (1,1), multiplies, then squeezes to (18,).
    # TF requires explicit expansion for this behavior.
    var_node_1_exp = tf.expand_dims(var_node_1, 0) # (1, 1)
    var_node_0 = tf.squeeze(tf.matmul(var_node_1_exp, var_node_5), 0) # (18,)

    # result = var_node_0 * sentinel
    result = var_node_0 * sentinel

    # if result.is_complex(): result = result.real
    # Conversion: Check dtype and take real part if complex.
    if result.dtype.is_complex:
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# Conversion: torch.tensor(1.0, requires_grad=True) -> tf.constant (or tf.Variable if gradients were used).
# Defined as float64 to match PyTorch's promotion behavior with var_node_0.
sentinel = tf.constant(1.0, dtype=tf.float64)

# arg_0 = torch.as_strided(torch.randn(20).to(torch.float64), (2, 10), (10, 1))
# Conversion: torch.randn -> tf.random.normal. as_strided with standard stride -> tf.reshape.
arg_0 = tf.reshape(tf.random.normal((20,), dtype=tf.float64), (2, 10))

# arg_1 = torch.as_strided(torch.randn(30).to(torch.float64), (10, 3), (3, 1))
arg_1 = tf.reshape(tf.random.normal((30,), dtype=tf.float64), (10, 3))

args = (arg_0, arg_1, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Conversion: tf.function is the TF equivalent of torch.compile.
compiled_program = tf.function(fuzzed_program)

result_compiled = compiled_program(*args)
print('✅ compile success')
```