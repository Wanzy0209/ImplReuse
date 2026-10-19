```python
import tensorflow as tf

# PyTorch specific configs ignored (torch._dynamo)
# TensorFlow uses tf.function for graph compilation

# Set random seed
tf.random.set_seed(1014698)

def fuzzed_program(arg_0, sentinel):
    # var_node_1 = arg_0
    var_node_1 = arg_0

    # var_node_3 = torch.full((), True, dtype=torch.bool)
    var_node_3 = tf.constant(True, dtype=tf.bool)

    # _x_nz = torch.zeros((), dtype=torch.bool, device=var_node_3.device)
    # TF handles device placement implicitly or via tf.device contexts.
    _x_nz = tf.constant(False, dtype=tf.bool)

    # _x_nz_flat = _x_nz.reshape(-1)
    _x_nz_flat = tf.reshape(_x_nz, [-1])

    # _x_nz_flat[:20] = True
    # PyTorch in-place assignment. TF tensors are immutable.
    # We simulate the result: a tensor of size 20 filled with True.
    # Note: The original PyTorch code would likely raise an IndexError on a scalar tensor,
    # but we translate the intent of creating a sequence of True values.
    _x_nz_flat = tf.ones([20], dtype=tf.bool)

    # var_node_2 = torch.nonzero(_x_nz)
    # Note: The original code uses _x_nz (scalar) but modified it via the view.
    # We use the modified tensor _x_nz_flat.
    # tf.where returns indices of non-zero/True elements.
    var_node_2 = tf.where(_x_nz_flat)

    # var_node_0 = torch.add(var_node_1, var_node_2)
    var_node_0 = tf.add(var_node_1, var_node_2)

    # result = var_node_0 * sentinel
    result = var_node_0 * sentinel

    # if result.is_complex(): result = result.real
    # tf.math.real returns the real part. If real, returns the tensor.
    result = tf.math.real(result)

    return result

# Sentinel tensor to ensure gradient computation
# PyTorch: torch.tensor(1.0, requires_grad=True)
# TF: tf.Variable with trainable=True (default for float)
sentinel = tf.Variable(1.0, dtype=tf.float32)

# arg_0 = torch.as_strided(torch.randint(5, 30, (20,)).to(torch.int64), (20, 0), (1, 20))
# TF equivalent: create random ints, reshape to (20, 0).
# Strides are implicit in TF memory layout, but (20, 0) shape is valid.
base_tensor = tf.random.uniform((20,), minval=5, maxval=30, dtype=tf.int64)
arg_0 = tf.reshape(base_tensor, (20, 0))

args = (arg_0, sentinel)

# Run eager
result_original = fuzzed_program(*args)
print('✅ eager success')

# Compile
# torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```