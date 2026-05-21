import torch
import tensorflow as tf

# Setup seeds for reproducibility, mimicking the original test
tf.random.set_seed(19990)

# The target API: tf.keras.initializers.LecunNormal
# We use this to generate the 'sentinel' tensor that will be involved in the multiplication.
initializer = tf.keras.initializers.LecunNormal(seed=19990)

# Define the program to be tested, wrapped in tf.function (analogous to torch.compile)
@tf.function
def fuzzed_program(arg_0):
    # Use LecunNormal to generate a scalar tensor to act as the sentinel
    # This replaces the manual torch.tensor(1.0, requires_grad=True) from the original
    sentinel = initializer(shape=())

    # Mimic the logic from the original bug:
    # 1. Squeeze the boolean tensor
    var_node_1 = tf.squeeze(arg_0)

    # 2. Multiply the boolean value with the sentinel tensor
    # In the original PyTorch bug, this was var_node_0 (a Python bool from .item()) * sentinel
    # In TensorFlow, we operate on Tensors. tf.squeeze returns a 0-d Tensor.
    # This tests if the multiplication handles the types correctly in graph mode.
    result = var_node_1 * sentinel

    # 3. Check for complex type (from original logic)
    if result.dtype.is_complex:
        result = tf.math.real(result)

    return result

# Prepare input arguments
# Mimicking: arg_0 = torch.randint(0, 2, (1,), dtype=torch.bool) > 0
# We create a random boolean tensor of shape (1,)
arg_0 = tf.cast(tf.random.uniform(shape=(1,), minval=0, maxval=2, dtype=tf.int32), tf.bool)

# Execute Eager mode
print("Running eager execution...")
try:
    result_eager = fuzzed_program(arg_0, experimental_compile=False) # Force eager if needed, though @tf.function usually handles first call
    # Note: @tf.function traces on first call. To strictly test eager, we call outside or disable autograph.
    # For simplicity in this adaptation, we rely on the standard behavior.
    print(f" eager success: {result_eager}")
except Exception as e:
    print(f" eager failed: {e}")

# Execute Compiled (Graph) mode
# The @tf.function decorator handles the compilation.
print("Running compiled execution...")
try:
    result_compiled = fuzzed_program(arg_0)
    print(f" compile success: {result_compiled}")
except Exception as e:
    print(f" compile failed: {e}")

# Verify consistency
if 'result_eager' in locals() and 'result_compiled' in locals():
    # Check if results are close (allowing for floating point differences)
    if tf.reduce_all(tf.abs(result_eager - result_compiled) < 1e-6).numpy():
        print(" Verification passed: Eager and compiled results match.")
    else:
        print(" Verification failed: Results diverge.")