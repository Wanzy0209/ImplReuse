import torch
import tensorflow as tf

# Reproduce the core logic: Handling zero-sized dimensions (20, 0)
# in both Eager and Compiled (Graph) modes.

# Set seed for reproducibility
tf.random.set_seed(1014698)

# The shape that caused issues in the original PyTorch bug
target_shape = (20, 0)

# Initialize the similar API
initializer = tf.keras.initializers.LecunUniform(seed=1014698)

# 1. Test in Eager Mode
print("Testing Eager Mode...")
try:
    result_eager = initializer(shape=target_shape)
    print(f" Eager success. Output shape: {result_eager.shape}, dtype: {result_eager.dtype}")
except Exception as e:
    print(f" Eager failed: {e}")

# 2. Test in Compiled (Graph) Mode
# tf.function is the TensorFlow equivalent of torch.compile
print("Testing Compiled (Graph) Mode...")
@tf.function
def compiled_initializer(shape):
    return initializer(shape=shape)

try:
    result_compiled = compiled_initializer(target_shape)
    print(f" Compile success. Output shape: {result_compiled.shape}, dtype: {result_compiled.dtype}")
except Exception as e:
    print(f" Compile failed: {e}")

# 3. Verify Consistency
# Check if both modes produced the same shape and dtype
if 'result_eager' in locals() and 'result_compiled' in locals():
    assert result_eager.shape == result_compiled.shape, "Shape mismatch between eager and compiled"
    assert result_eager.dtype == result_compiled.dtype, "Dtype mismatch between eager and compiled"
    print(" Consistency check passed: Eager and Compiled modes match.")