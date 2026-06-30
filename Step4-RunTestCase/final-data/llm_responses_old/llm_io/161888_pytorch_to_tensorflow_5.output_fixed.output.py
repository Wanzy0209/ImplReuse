import tensorflow as tf
import traceback

print("TensorFlow Version:", tf.__version__)

# Recreate tensors similar to the original PyTorch bug report
# Fix: tf.random.uniform in TF 1.x does not support int16, use int32
tensor1 = tf.random.uniform((9, 3, 7), minval=-100, maxval=100, dtype=tf.int32)

# Fix: tf.random.uniform in TF 1.x does not support bool directly, generate int32 and cast
tensor2 = tf.cast(tf.random.uniform((1, 6, 4, 8), minval=0, maxval=2, dtype=tf.int32), tf.bool)

# The original bug involved passing a list of tensors and a huge integer
huge_int = 154691921484029491302139942063978250367

# Test Case 1: Passing a list of tensors as the input tensor
# Corresponds to r1(*input[2]) where input[2] = [tensor1, tensor2] in the original bug
print("\n--- Test Case 1: Invalid input type (list of tensors) ---")
try:
    # tf.strided_slice expects a single tensor as input_, not a list
    result = tf.strided_slice(input_=[tensor1, tensor2], begin=[0, 0, 0], end=[1, 1, 1])
    print("Result:", result)
except Exception as e:
    print(f"Caught Exception: {type(e).__name__}")
    # Uncomment to see full traceback
    # print(traceback.format_exc())

# Test Case 2: Passing a huge integer as indices
# Corresponds to the constructor argument in the original bug
print("\n--- Test Case 2: Invalid index value (huge integer) ---")
try:
    # tf.strided_slice expects indices compatible with the tensor rank
    result = tf.strided_slice(input_=tensor1, begin=huge_int, end=huge_int)
    print("Result:", result)
except Exception as e:
    print(f"Caught Exception: {type(e).__name__}")
    # Uncomment to see full traceback
    # print(traceback.format_exc())