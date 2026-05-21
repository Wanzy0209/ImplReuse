import torch
import tensorflow as tf
import numpy as np

# Setup seeds to match the original test context
tf.random.set_seed(19990)
np.random.seed(19990)

def fuzzed_program(arg_0):
    # arg_0: size=(1,), dtype=bool
    # Mimic the squeeze operation from the original test
    var_node_1 = tf.squeeze(arg_0)  # size=(), dtype=bool

    # Use the Similar API: tf.keras.initializers.HeNormal
    # to generate the tensor to be multiplied, replacing the manual sentinel creation.
    # This tests the interaction between the initializer and the graph logic.
    initializer = tf.keras.initializers.HeNormal(seed=19990)
    sentinel = initializer(shape=())  # size=(), dtype=float32

    # The core logic from the bug: multiplication
    # In PyTorch, this was var_node_0 (Python bool) * sentinel (Tensor).
    # In TensorFlow, we use the 0-d tensor var_node_1 directly to ensure graph compatibility.
    result = var_node_1 * sentinel

    # Original check for complex numbers (HeNormal produces floats, so this is safe)
    if result.dtype == tf.complex64 or result.dtype == tf.complex128:
        result = tf.math.real(result)

    return result

# Input arguments
# Original: torch.randint(0, 2, (1,), dtype=torch.bool) > 0
# TF equivalent:
arg_0 = tf.constant([True], dtype=tf.bool)

# 1. Eager Execution
print("Running eager execution...")
try:
    result_eager = fuzzed_program(arg_0)
    print(f" eager success: {result_eager.numpy()}")
except Exception as e:
    print(f" eager failed: {e}")

# 2. Compiled Execution (tf.function equivalent to torch.compile)
print("Running compiled execution...")
try:
    compiled_program = tf.function(fuzzed_program)
    result_compiled = compiled_program(arg_0)
    print(f" compile success: {result_compiled.numpy()}")
except Exception as e:
    print(f" compile failed: {e}")

# 3. Verification
# Check if results are close (HeNormal with same seed should produce same values)
if 'result_eager' in locals() and 'result_compiled' in locals():
    if np.allclose(result_eager.numpy(), result_compiled.numpy()):
        print(" Results match.")
    else:
        print(" Results diverge.")