```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# TensorFlow (tf.function) handles scalar outputs automatically, no equivalent config needed.

# torch.manual_seed(238)
tf.random.set_seed(238)

def fuzzed_program(arg_0, sentinel):
    # var_node_2 = torch.full((), 1, dtype=torch.int16)
    var_node_2 = tf.constant(1, dtype=tf.int16, shape=()) # size=(), dtype=int16

    # var_node_3 = arg_0
    var_node_3 = arg_0

    # var_node_1 = torch.add(var_node_2, var_node_3)
    var_node_1 = tf.add(var_node_2, var_node_3)

    # var_node_5 = torch.full((1,), 3, dtype=torch.int16)
    var_node_5 = tf.constant([3], dtype=tf.int16) # size=(1,), dtype=int16

    # var_node_4 = torch.squeeze(var_node_5)
    var_node_4 = tf.squeeze(var_node_5) # size=(), dtype=int16

    # var_node_0 = torch.div(var_node_1, var_node_4)
    # Note: torch.div with integers performs true division (result is float)
    var_node_0 = tf.math.divide(var_node_1, var_node_4) # size=(), dtype=float

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel

    # if result.is_complex():
    if result.dtype.is_complex:
        # result = result.real
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# torch.tensor(1.0, requires_grad=True)
sentinel = tf.Variable(1.0, dtype=tf.float32)

# arg_0 = torch.as_strided(torch.randn(1).to(torch.int16), (), ())
# torch.randn(1) -> tf.random.normal([1])
# .to(torch.int16) -> tf.cast(..., tf.int16)
# as_strided(..., (), ()) -> tf.reshape(..., []) (creates a scalar view)
arg_0 = tf.reshape(tf.cast(tf.random.normal([1]), tf.int16), [])

args = (arg_0, sentinel)

# Eager execution
# PyTorch computes gradients via .backward(). TF uses GradientTape.
with tf.GradientTape() as tape:
    out_eager = fuzzed_program(*args)
# Mimic .sum().backward() to ensure graph connectivity
grads_eager = tape.gradient(tf.reduce_sum(out_eager), sentinel)
print('Eager Success! ✅')

# Compile program
# torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)

# Compiled execution
with tf.GradientTape() as tape:
    out_compiled = compiled_program(*args)
# Mimic .sum().backward()
grads_compiled = tape.gradient(tf.reduce_sum(out_compiled), sentinel)
print('Compile Success! ✅')

# Comparison logic
# out_eager_sum = out_eager.sum()
out_eager_sum = tf.reduce_sum(out_eager)
# out_compiled_sum = out_compiled.sum()
out_compiled_sum = tf.reduce_sum(out_compiled)

# diff = (out_eager_sum - out_compiled_sum).abs().item()
diff = tf.abs(out_eager_sum - out_compiled_sum).numpy()
# rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100
rel_diff = diff / (tf.abs(out_eager_sum).numpy() + 1e-12) * 100

print(f'Relative diff (sum): {rel_diff:.6f}%')
if rel_diff > 5 and diff > 1:
    print(f'❌ Forward output sums differ significantly (relative and absolute)!')
    # print('out_eager_sum:', out_eager_sum.item())
    print('out_eager_sum:', out_eager_sum.numpy())
    # print('out_compiled_sum:', out_compiled_sum.item())
    print('out_compiled_sum:', out_compiled_sum.numpy())
    print('Absolute diff:', diff)
    print('Relative diff (%):', rel_diff)
    import sys; sys.exit(1)
```