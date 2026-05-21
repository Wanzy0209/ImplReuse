import torch
import tensorflow as tf

print("TensorFlow Version:", tf.__version__, flush=True)

# Recreate the inputs from the PyTorch bug report
# PyTorch: torch.empty((5, 7, 4, 3, 7, 6), dtype=torch.int8)
input_tensor = tf.empty((5, 7, 4, 3, 7, 6), dtype=tf.int8)

# PyTorch: torch.empty((4, 9, 2), dtype=torch.int32)
# In the original bug, this was passed as the 'indices' argument.
# For tf.keras.ops.moveaxis, this maps to the 'source' argument.
indices_tensor = tf.empty((4, 9, 2), dtype=tf.int32)

# PyTorch: ()
# In the original bug, this was passed as the 'output_size' argument.
# For tf.keras.ops.moveaxis, this maps to the 'destination' argument.
output_size = ()

# PyTorch: False
# This boolean flag does not have a direct equivalent in the moveaxis signature, so it is omitted.

# Call the similar API: tf.keras.ops.moveaxis
# Mapping: input -> a, indices -> source, output_size -> destination
# Note: The original PyTorch bug involved passing a tensor as 'indices' where specific shapes were expected.
# Here we pass a tensor as 'source' (which expects an int or tuple) to test robustness.
try:
    result = tf.keras.ops.moveaxis(input_tensor, indices_tensor, output_size)
    print("Result:", result)
except Exception as e:
    print(f"Exception caught: {type(e).__name__}: {e}")